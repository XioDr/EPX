import openpyxl
import os
import glob
import re # For parsing day numbers from headers

# --- Configuration ---
input_directory = 'APX-EPEX Prijzen'
output_file = 'merged_apx_epex_openpyxl_filtered_padded.xlsx' # Updated output name
header_row_number = 3
data_start_row_number = 4
# --- End Configuration ---

# Define the *complete* set of columns for the output file
# Ensures consistency regardless of the number of days in source files
final_header = ["Start", "Eind"] + [str(i) for i in range(1, 32)] + ["total", "week", "weekend"]
# Create a mapping for quick lookup of output column index
final_header_map = {name: idx for idx, name in enumerate(final_header)}

# Create a new workbook for the output
wb_out = openpyxl.Workbook()
sheet_out = wb_out.active
sheet_out.title = "Merged Data"

# Write the final, complete header row immediately
sheet_out.append(final_header)
print(f"Output header written: {final_header}")

# Find all XLSX files in the input directory
search_pattern = os.path.join(input_directory, '*.xlsx')
xlsx_files = glob.glob(search_pattern)

if not xlsx_files:
    print(f"Error: No XLSX files found matching pattern '{search_pattern}'.")
else:
    print(f"Found {len(xlsx_files)} XLSX files to process...")

    total_rows_appended = 0
    for filename in xlsx_files:
        print(f"Processing '{filename}'...")
        file_rows_appended = 0
        try:
            wb_in = openpyxl.load_workbook(filename, data_only=True, read_only=True)
            sheet_in = wb_in.active

            if sheet_in.max_row < header_row_number:
                print(f"Warning: File '{filename}' too short. Skipping.")
                wb_in.close()
                continue

            # Read the header row specific to *this* input file
            input_header_cells = sheet_in[header_row_number]
            input_header = [str(cell.value).strip() if cell.value is not None else '' for cell in input_header_cells]
            # Create a map from this file's header name to its column index (0-based)
            input_header_map = {name: idx for idx, name in enumerate(input_header) if name} # Ignore empty header names

            if not input_header or input_header[0].lower() != 'start':
                 print(f"Warning: Invalid header found in '{filename}'. Skipping file.")
                 wb_in.close()
                 continue

            # Iterate through data rows
            for row_idx in range(data_start_row_number, sheet_in.max_row + 1):
                input_row_cells = sheet_in[row_idx]
                input_row_values = [cell.value for cell in input_row_cells] # Keep None for empty cells initially

                # Filter row based on first column (e.g., must contain ':')
                if not (input_row_values and input_row_values[0] is not None and ':' in str(input_row_values[0])):
                    continue # Skip if first cell doesn't look like a time

                # Create the output row, initialized with None for all final columns
                output_row = [None] * len(final_header)

                # Map values from input row to output row based on headers
                for input_col_name, input_col_idx in input_header_map.items():
                    # Ensure we don't read beyond the actual length of the input row
                    if input_col_idx < len(input_row_values):
                        value = input_row_values[input_col_idx]
                    else:
                        value = None # Treat as missing if input row is shorter than header indicated

                    # Check if it's a standard column name
                    if input_col_name in final_header_map:
                        output_idx = final_header_map[input_col_name]
                        output_row[output_idx] = value
                    else:
                        # Check if it's a date column (e.g., "DD-MM")
                        match = re.match(r"(\d{1,2})[\-\/]", input_col_name)
                        if match:
                            try:
                                day_num_str = str(int(match.group(1)))
                                # Check if this day number ('1' to '31') is in our final header map
                                if day_num_str in final_header_map:
                                    output_idx = final_header_map[day_num_str]
                                    output_row[output_idx] = value
                            except (ValueError, IndexError):
                                pass # Ignore if not a valid day number format

                # Only append if the row seems to have actual data (beyond just time)
                if any(val is not None for val in output_row[1:]): # Check from second column onwards
                    sheet_out.append(output_row)
                    file_rows_appended += 1

            print(f"-> Appended {file_rows_appended} data rows (after filtering).")
            total_rows_appended += file_rows_appended
            wb_in.close()

        except FileNotFoundError:
            print(f"Error: File not found - {filename}")
        except Exception as e:
            print(f"An error occurred processing {filename}: {type(e).__name__} - {e}")
            if 'wb_in' in locals() and wb_in:
                 try: wb_in.close()
                 except: pass

    # --- Save the final merged workbook ---
    if total_rows_appended > 0: # Check if any data rows were appended across all files
        try:
            print(f"\nAttempting to save merged data to '{output_file}'...")
            wb_out.save(output_file)
            print(f"Successfully merged data into '{output_file}'.")
            print(f"Total data rows appended: {total_rows_appended}")
            print(f"Total rows in output file (including header): {sheet_out.max_row}")
        except Exception as e:
             print(f"An error occurred writing the output file '{output_file}': {type(e).__name__} - {e}")
    elif not xlsx_files:
        pass
    else:
        print("\nNo valid data rows were found or processed. Output file not created.")

