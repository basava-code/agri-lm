package com.agri;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVPrinter;
import org.apache.commons.csv.CSVRecord;

import java.io.*;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Queue;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;

public class LlmNoiseFilter {

    private static final String VLLM_BASE_URL = "http://localhost:8000/v1/chat/completions";
    private static final String VLLM_MODEL = "./qwen-2.5-3b-instruct";
    
    private static final String INPUT_DIR = "raw_data";
    private static final String OUTPUT_DIR = "classified_data";
    private static final String DISCARD_DIR = "discarded";
    
    // We can define massive concurrency with Virtual Threads
    private static final int MAX_CONCURRENT_REQUESTS = 5000;
    
    private static final ObjectMapper mapper = new ObjectMapper();
    private static final HttpClient client = HttpClient.newBuilder()
            .executor(Executors.newVirtualThreadPerTaskExecutor())
            .version(HttpClient.Version.HTTP_1_1)
            .build();

    private static final String SYSTEM_PROMPT = 
        "You are an expert agricultural dataset filter. Your ONLY task is to classify " +
        "whether the given farmer query asks for actionable farming advice, crop techniques, " +
        "pest management, weather-crop mitigation, or livestock care.\n\n" +
        "Follow these rules strictly:\n" +
        "- Reply with EXACTLY the word 'KEEP' if it contains useful agricultural information.\n" +
        "- Reply with EXACTLY the word 'DISCARD' if it is a general weather report, " +
        "an administrative redirection (e.g. 'contact Patwari/Company'), a query about money/schemes, " +
        "market rates (mandi bhav), or empty/gibberish text.\n" +
        "- Do NOT output any preamble, markdown, punctuation, or extra words. Output only KEEP or DISCARD.";

    public static void main(String[] args) throws Exception {
        Files.createDirectories(Paths.get(OUTPUT_DIR));
        Files.createDirectories(Paths.get(DISCARD_DIR));

        System.out.println("Starting Java Virtual Threads Classification Pipeline...");
        System.out.println("Using JDK 21+ Virtual Threads with Concurrency Limit: " + MAX_CONCURRENT_REQUESTS);
        
        File inputDir = new File(INPUT_DIR);
        File[] csvFiles = inputDir.listFiles((dir, name) -> name.endsWith(".csv"));
        
        if (csvFiles == null || csvFiles.length == 0) {
            System.out.println("No CSV files found in 'raw_data/'.");
            return;
        }

        // Shared executor for submitting tasks
        try (ExecutorService virtualExecutor = Executors.newVirtualThreadPerTaskExecutor()) {
            // We use a semaphore to prevent loading millions of rows into memory faster than the API can handle
            Semaphore throttle = new Semaphore(MAX_CONCURRENT_REQUESTS);

            for (File file : csvFiles) {
                processFile(file, virtualExecutor, throttle);
            }
        }
        
        System.out.println("\nAll processing complete!");
    }

    private static void processFile(File inputFile, ExecutorService executor, Semaphore throttle) throws Exception {
        System.out.println("\nProcessing " + inputFile.getName() + "...");
        
        Path outputPath = Paths.get(OUTPUT_DIR, inputFile.getName());
        Path discardPath = Paths.get(DISCARD_DIR, inputFile.getName());
        
        if (Files.exists(outputPath) && Files.exists(discardPath)) {
            System.out.println("Output files exist. Skipping " + inputFile.getName());
            return;
        }

        Queue<CSVRecord> keepRecords = new ConcurrentLinkedQueue<>();
        Queue<CSVRecord> discardRecords = new ConcurrentLinkedQueue<>();
        
        List<String> headers = new ArrayList<>();
        
        try (Reader reader = new FileReader(inputFile);
             CSVParser parser = new CSVParser(reader, CSVFormat.DEFAULT.withFirstRecordAsHeader().withIgnoreHeaderCase().withTrim())) {
            
            headers.addAll(parser.getHeaderNames());
            
            List<Future<Void>> futures = new ArrayList<>();
            AtomicInteger processedCount = new AtomicInteger(0);

            for (CSVRecord record : parser) {
                throttle.acquire();
                
                futures.add(executor.submit(() -> {
                    try {
                        String query = record.isMapped("QueryText") ? record.get("QueryText") : "";
                        String decision = classifyQuery(query);
                        
                        if ("DISCARD".equals(decision)) {
                            discardRecords.add(record);
                        } else {
                            keepRecords.add(record);
                        }
                        
                        int count = processedCount.incrementAndGet();
                        if (count % 1000 == 0) {
                            System.out.println("Processed " + count + " rows...");
                        }
                    } catch (Exception e) {
                        System.err.println("Error processing record: " + e.getMessage());
                        keepRecords.add(record); // Default to keep on fail
                    } finally {
                        throttle.release();
                    }
                    return null;
                }));
            }
            
            // Wait for all rows in this file to finish
            for (Future<Void> future : futures) {
                future.get();
            }
            
            System.out.println("Saving " + inputFile.getName() + " (Kept: " + keepRecords.size() + ", Discarded: " + discardRecords.size() + ")");
            writeRecords(outputPath, headers, keepRecords);
            writeRecords(discardPath, headers, discardRecords);
        }
    }

    private static String classifyQuery(String query) throws Exception {
        if (query == null) query = "";
        if (query.length() > 3000) {
            query = query.substring(0, 3000) + "...";
        }

        ObjectNode rootNode = mapper.createObjectNode();
        rootNode.put("model", VLLM_MODEL);
        rootNode.put("max_tokens", 2);
        rootNode.put("temperature", 0.0);

        ArrayNode messagesNode = rootNode.putArray("messages");
        
        ObjectNode systemMsg = mapper.createObjectNode();
        systemMsg.put("role", "system");
        systemMsg.put("content", SYSTEM_PROMPT);
        messagesNode.add(systemMsg);
        
        ObjectNode userMsg = mapper.createObjectNode();
        userMsg.put("role", "user");
        userMsg.put("content", "Farmer Query: " + query);
        messagesNode.add(userMsg);

        String jsonBody = mapper.writeValueAsString(rootNode);

        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(VLLM_BASE_URL))
                .header("Content-Type", "application/json")
                .header("Authorization", "Bearer vllm-does-not-care")
                .POST(HttpRequest.BodyPublishers.ofString(jsonBody))
                .build();

        int retries = 3;
        for (int i = 0; i < retries; i++) {
            try {
                HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());
                if (response.statusCode() == 400) {
                    System.err.println("400 Bad Request: " + response.body());
                    return "KEEP";
                }
                
                JsonNode responseNode = mapper.readTree(response.body());
                String decision = responseNode.path("choices").get(0).path("message").path("content").asText().trim().toUpperCase();
                
                if (decision.contains("DISCARD")) {
                    return "DISCARD";
                } else {
                    return "KEEP";
                }
            } catch (Exception e) {
                if (i == retries - 1) {
                    return "KEEP";
                }
                Thread.sleep(1000);
            }
        }
        return "KEEP";
    }
    
    private static void writeRecords(Path path, List<String> headers, Queue<CSVRecord> records) throws IOException {
        if (records.isEmpty()) return;
        
        try (Writer writer = new FileWriter(path.toFile());
             CSVPrinter printer = new CSVPrinter(writer, CSVFormat.DEFAULT.withHeader(headers.toArray(new String[0])))) {
             
             for (CSVRecord record : records) {
                 List<String> values = new ArrayList<>();
                 for (String header : headers) {
                     values.add(record.isMapped(header) ? record.get(header) : "");
                 }
                 printer.printRecord(values);
             }
        }
    }
}
