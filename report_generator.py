import pandas as pd
from io import BytesIO
import xlsxwriter
from fpdf import FPDF

# Function to generate CSV
def generate_csv(dataframe):
    return dataframe.to_csv(index=False).encode('utf-8')

# Function to generate Excel
def generate_excel(dataframe):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    dataframe.to_excel(writer, sheet_name='Sheet1', index=False)
    writer.close()
    return output.getvalue()

# Function to generate PDF
def generate_pdf(dataframe, selected_columns):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Report", ln=True, align="C")
    
    # Add table headers
    col_width = pdf.w / len(selected_columns) - 1
    for column in selected_columns:
        pdf.cell(col_width, 10, column, border=1)
    pdf.ln()
    
    # Add data rows
    for row in dataframe.itertuples():
        for item in row[1:]:
            pdf.cell(col_width, 10, str(item), border=1)
        pdf.ln()

    return pdf.output(dest='S').encode('latin1')
