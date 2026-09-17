# KCC Telephonic Noise Filter (Java 21 Project Loom)

High-throughput, asynchronous telephonic transcript cleaner and noise classifier powered by Java 21 Virtual Threads (`Project Loom`).

## Architecture
- **Executor:** Non-blocking `Executors.newVirtualThreadPerTaskExecutor()` handling up to 5,000 concurrent HTTP requests.
- **Backend:** Evaluates records against a local vLLM endpoint (`Qwen-2.5-3B-Instruct` or `Gemma-4-E2B-it`).
- **Classification:** Segregates raw Kisan Call Center records into valid agronomic dialogues and discarded greeting/telephonic noise.

## Build & Run

```bash
# Compile and build shaded fat-JAR
mvn clean package -DskipTests

# Run against raw data directory
java -jar target/java-filter-1.0-SNAPSHOT.jar
```
