import torch

def test_model():
    # Define the log file path
    log_dir = "log.txt"
    
    # Open the log file and read the last line
    with open(log_dir, 'r') as file:
        lines = file.readlines()
        last_line = lines[-1] if lines else None

    # Ensure we have a valid last line
    if last_line:
        # Extract the loss value from the last line
        try:
            loss_value = float(last_line.split()[-1])
            assert loss_value < 1.0, "Training loss is not below the threshold."
            print("Training test passed. Final loss:", loss_value)
        except ValueError:
            print("Could not parse loss value from the log file.")
    else:
        print("Log file is empty or does not exist.")

if __name__ == "__main__":
    test_model()
