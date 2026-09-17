import os
import glob
import pandas as pd

def remove_duplicates(input_dir="raw_data"):
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {input_dir}/")
        return
        
    total_removed = 0
    total_initial = 0
    
    print(f"Starting duplicate removal for CSVs in '{input_dir}'...\n")
    
    for file_path in csv_files:
        filename = os.path.basename(file_path)
        print(f"Processing {filename}...")
        
        # Read the CSV with the same robust settings as your main pipeline
        try:
            df = pd.read_csv(file_path, low_memory=False, encoding_errors='replace')
        except Exception as e:
            print(f" -> Error reading {filename}: {e}")
            continue
            
        initial_rows = len(df)
        total_initial += initial_rows
        
        # Drop rows where BOTH the question and the answer are exactly identical
        # (Using just 'QueryText' could accidentally delete a follow-up with a different answer)
        df_cleaned = df.drop_duplicates(subset=['QueryText', 'KccAns']).dropna(subset=['QueryText', 'KccAns'])
        
        final_rows = len(df_cleaned)
        removed = initial_rows - final_rows
        total_removed += removed
        
        if removed > 0:
            print(f" -> Removed {removed} duplicate rows ({(removed/initial_rows)*100:.2f}% of file).")
            # Save back to the exact same file, preserving the original structure
            df_cleaned.to_csv(file_path, index=False)
        else:
            print(f" -> No duplicates found.")
            
    print("-" * 50)
    print(f"Cleanup Complete!")
    print(f"Total rows scanned: {total_initial}")
    print(f"Total duplicates destroyed: {total_removed}")
    
    if total_initial > 0:
        print(f"Overall reduction: {(total_removed/total_initial)*100:.2f}%")

if __name__ == "__main__":
    # Ensure this points to the exact same 'raw_data' folder your main pipeline uses
    remove_duplicates("raw_data")
