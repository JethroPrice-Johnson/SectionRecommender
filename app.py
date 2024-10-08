import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import tkinter as tk
from tkinter import messagebox

# Load data (Adjust paths as necessary)
data_manufacturer_1 = pd.read_csv('./data/Futura_WuSag.csv')
data_manufacturer_2 = pd.read_csv('./data/DHS_WuSag.csv')

# Global variables for trained models and accuracies
model_1 = None
model_2 = None
accuracy_1 = 0
accuracy_2 = 0

# Train models using full data sets
def train_models():
    global model_1, model_2, accuracy_1, accuracy_2

    # Prepare features and labels for the full datasets
    features_1 = data_manufacturer_1[['UDL Capacity', 'Length']]
    labels_1 = data_manufacturer_1['Section Code']

    features_2 = data_manufacturer_2[['UDL Capacity', 'Length']]
    labels_2 = data_manufacturer_2['Section Code']

    # Train models with full data
    model_1 = RandomForestClassifier()
    model_1.fit(features_1, labels_1)

    model_2 = RandomForestClassifier()
    model_2.fit(features_2, labels_2)

    # Calculate refined accuracy based on cost-minimisation strategy
    accuracy_1 = refine_accuracy(model_1, features_1, data_manufacturer_1)
    accuracy_2 = refine_accuracy(model_2, features_2, data_manufacturer_2)

    # Model training completed
    messagebox.showinfo("Model Training", "Models for both manufacturers have been trained successfully.")

# Refine predictions based on cost and calculate custom accuracy
def refine_accuracy(model, features, full_data):
    # Make predictions using the model
    predicted_sections = model.predict(features)
    
    # Add predicted sections to the full data
    full_data['Predicted Section'] = predicted_sections

    # Calculate accuracy based on lowest cost strategy
    correct_count = 0
    for i, row in full_data.iterrows():
        # Get sections with similar UDL and Length
        similar_sections = full_data[(full_data['UDL Capacity'] == row['UDL Capacity']) &
                                     (full_data['Length'] == row['Length'])]
        
        # Identify the lowest cost section among similar options
        optimal_section = similar_sections.loc[similar_sections['Cost'].idxmin()]['Section Code']
        
        # Check if the predicted section is the same as the optimal section
        if row['Predicted Section'] == optimal_section:
            correct_count += 1

    # Return refined accuracy
    return correct_count / len(full_data)

# Find and display predictions and optimal sections
def find_and_display_sections():
    try:
        # Get user inputs
        udl_input = float(udl_entry.get())
        length_input = float(length_entry.get())
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numeric values for UDL and Length.")
        return

    # Filter the data based on user inputs
    filtered_data_1 = data_manufacturer_1[(data_manufacturer_1['UDL Capacity'] >= udl_input) & (data_manufacturer_1['Length'] >= length_input)].copy()
    filtered_data_2 = data_manufacturer_2[(data_manufacturer_2['UDL Capacity'] >= udl_input) & (data_manufacturer_2['Length'] >= length_input)].copy()

    # Clear previous results
    clear_labels()

    optimal_section_1 = None
    optimal_section_2 = None

    # Use the trained models to predict section codes for the filtered data
    if model_1 is not None and not filtered_data_1.empty:
        filtered_data_1['Predicted Section'] = model_1.predict(filtered_data_1[['UDL Capacity', 'Length']])
        optimal_section_1 = filtered_data_1.loc[filtered_data_1['Cost'].idxmin()] if not filtered_data_1.empty else None

        # Display predicted and optimal details for Manufacturer 1
        display_section_details(filtered_data_1.iloc[0], 3, 0, "Predicted")
        accuracy_label_1.config(text=f"Accuracy: {accuracy_1 * 100:.2f}%")
        if optimal_section_1 is not None:
            display_section_details(optimal_section_1, 3, 1, "Optimal")

    if model_2 is not None and not filtered_data_2.empty:
        filtered_data_2['Predicted Section'] = model_2.predict(filtered_data_2[['UDL Capacity', 'Length']])
        optimal_section_2 = filtered_data_2.loc[filtered_data_2['Cost'].idxmin()] if not filtered_data_2.empty else None

        # Display predicted and optimal details for Manufacturer 2
        display_section_details(filtered_data_2.iloc[0], 3, 2, "Predicted")
        accuracy_label_2.config(text=f"Accuracy: {accuracy_2 * 100:.2f}%")
        if optimal_section_2 is not None:
            display_section_details(optimal_section_2, 3, 3, "Optimal")

    # Determine and display the most cost-effective manufacturer in green
    if optimal_section_1 is not None and optimal_section_2 is not None:
        if optimal_section_1['Cost'] < optimal_section_2['Cost']:
            praise_label.config(text="Manufacturer 1 offers the most cost-effective option.", fg="green")
        else:
            praise_label.config(text="Manufacturer 2 offers the most cost-effective option.", fg="green")
    elif optimal_section_1 is not None:
        praise_label.config(text="Only Manufacturer 1 offers a valid section.", fg="green")
    elif optimal_section_2 is not None:
        praise_label.config(text="Only Manufacturer 2 offers a valid section.", fg="green")
    else:
        praise_label.config(text="No valid sections available from either manufacturer.", fg="red")

# Clear all output labels
def clear_labels():
    for label in output_labels:
        label.config(text="")
    accuracy_label_1.config(text="")
    accuracy_label_2.config(text="")
    praise_label.config(text="")

# Display the details of a section in the GUI
def display_section_details(section, row, column, section_type):
    details = f"Section: {section['Section Code']}\n" \
              f"UDL: {section['UDL Capacity']}\n" \
              f"Length: {section['Length']}\n" \
              f"Cost: {section['Cost']}"
    
    output_labels[column].config(text=details)

# Create main window
root = tk.Tk()
root.title("Structural Section Optimisation")

# UDL and Length Inputs
udl_label = tk.Label(root, text="UDL Input")
udl_label.grid(row=0, column=0)
udl_entry = tk.Entry(root)
udl_entry.grid(row=0, column=1)

length_label = tk.Label(root, text="Length Input")
length_label.grid(row=0, column=2)
length_entry = tk.Entry(root)
length_entry.grid(row=0, column=3)

# Manufacturer 1 Output Headers
manufacturer_label_1 = tk.Label(root, text="Manufacturer 1", font=("Arial", 14, "bold"))
manufacturer_label_1.grid(row=1, column=0, columnspan=2)

# Manufacturer 2 Output Headers
manufacturer_label_2 = tk.Label(root, text="Manufacturer 2", font=("Arial", 14, "bold"))
manufacturer_label_2.grid(row=1, column=2, columnspan=2)

# Column Headers
predicted_label_1 = tk.Label(root, text="Predicted", font=("Arial", 12, "underline"))
predicted_label_1.grid(row=2, column=0)

optimal_label_1 = tk.Label(root, text="Optimal", font=("Arial", 12, "underline"))
optimal_label_1.grid(row=2, column=1)

predicted_label_2 = tk.Label(root, text="Predicted", font=("Arial", 12, "underline"))
predicted_label_2.grid(row=2, column=2)

optimal_label_2 = tk.Label(root, text="Optimal", font=("Arial", 12, "underline"))
optimal_label_2.grid(row=2, column=3)

# Create empty labels for displaying results
output_labels = [tk.Label(root, text="", justify='left') for _ in range(4)]
for idx, label in enumerate(output_labels):
    label.grid(row=3, column=idx, sticky='w')

# Accuracy labels below predicted sections
accuracy_label_1 = tk.Label(root, text="", justify='left', font=("Arial", 10, "italic"))
accuracy_label_1.grid(row=4, column=0)

accuracy_label_2 = tk.Label(root, text="", justify='left', font=("Arial", 10, "italic"))
accuracy_label_2.grid(row=4, column=2)

# Praise label for cost-effective manufacturer
praise_label = tk.Label(root, text="", justify='center', font=("Arial", 12, "bold"))
praise_label.grid(row=5, column=0, columnspan=4)

# Buttons for training and predicting
train_button = tk.Button(root, text="Train Models", command=train_models)
train_button.grid(row=6, column=0, columnspan=2)

predict_button = tk.Button(root, text="Find Optimal Sections", command=find_and_display_sections)
predict_button.grid(row=6, column=2, columnspan=2)

# Run the GUI loop
root.mainloop()
