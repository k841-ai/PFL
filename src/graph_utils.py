import os
import json
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import logging
import re
import base64
from io import BytesIO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('graph_utils.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class GraphGenerator:
    def __init__(self):
        self.graphs_dir = "graphs"
        os.makedirs(self.graphs_dir, exist_ok=True)
        logger.info(f"GraphGenerator initialized with directory: {self.graphs_dir}")

    def _save_graph(self, fig: go.Figure, graph_type: str) -> Tuple[str, str]:
        """Save graph as PNG and return file path and embed code."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{graph_type}_{timestamp}.png"
            filepath = os.path.join(self.graphs_dir, filename)
            
            logger.info(f"Attempting to save graph to: {filepath}")
            
            # Convert figure to PNG bytes
            img_bytes = fig.to_image(format="png")
            
            # Save the bytes to file
            with open(filepath, 'wb') as f:
                f.write(img_bytes)
            
            logger.info(f"Graph saved successfully at {filepath}")
            
            # Generate base64 encoded image for embed code
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')
            embed_code = f'<img src="data:image/png;base64,{img_base64}" alt="{graph_type} graph">'
            
            return filepath, embed_code
        except Exception as e:
            logger.error(f"Error saving graph: {str(e)}")
            # If saving fails, try to return just the base64 image
            try:
                img_bytes = fig.to_image(format="png")
                img_base64 = base64.b64encode(img_bytes).decode('utf-8')
                embed_code = f'<img src="data:image/png;base64,{img_base64}" alt="{graph_type} graph">'
                return None, embed_code
            except Exception as e2:
                logger.error(f"Error generating base64 image: {str(e2)}")
                raise

    def _extract_data_from_context(self, context: List[str], query: str) -> pd.DataFrame:
        """Extract relevant data from context based on query for Axis Finance and Mahindra Finance."""
        logger.info("Attempting to extract data from context for Axis Finance and Mahindra Finance")

        # Initialize empty data dictionaries for each company
        axis_data = {'Company': 'Axis Finance'}
        mahindra_data = {'Company': 'Mahindra Finance'}

        for i, doc_content in enumerate(context):
            logger.info(f"Processing document {i+1}/{len(context)}")

            # --- Extract Axis Finance data from a.rtf content ---
            # Check if the document content seems to be from a.rtf and is for Axis Finance FY2025
            if "Axis Finance Limited" in doc_content and ("March 31, 2025" in doc_content or "FY2025" in doc_content):
                logger.info("Found potential Axis Finance FY2025 data in context")

                # Extract Revenue from operations (looking for format like: Revenue from operations: 4,09,379.09 (2024: ...))
                revenue_match = re.search(r'Revenue from operations:\s*₹?([\d,]+\.\d+)\s*lakhs', doc_content)
                if revenue_match:
                    axis_data['Revenue from Operations (Lakhs ₹)'] = float(revenue_match.group(1).replace(',', ''))
                    logger.info(f"Extracted Axis Finance Revenue: {axis_data.get('Revenue from Operations (Lakhs ₹)', 'Not found')} lakhs")

                # Extract Profit After Tax (PAT) (looking for format like: Profit After Tax: 80,673.61 (2024: ...))
                pat_match = re.search(r'Profit After Tax:\s*₹?([\d,]+\.\d+)\s*lakhs', doc_content)
                if pat_match:
                    axis_data['Profit After Tax (Lakhs ₹)'] = float(pat_match.group(1).replace(',', ''))
                    logger.info(f"Extracted Axis Finance PAT: {axis_data.get('Profit After Tax (Lakhs ₹)', 'Not found')} lakhs")

                # Extract Total Financial Assets or Loans (net of ECL)
                # Looking for format like: Total Financial Assets: 39,28,095.42 (2024: ...)
                assets_match = re.search(r'Total Financial Assets:\s*₹?([\d,]+\.\d+)\s*lakhs', doc_content)
                if assets_match:
                    axis_data['Total Financial Assets (Lakhs ₹)'] = float(assets_match.group(1).replace(',', ''))
                    logger.info(f"Extracted Axis Finance Total Financial Assets: {axis_data.get('Total Financial Assets (Lakhs ₹)', 'Not found')} lakhs")
                else:
                     # Looking for format like: Loans (net of ECL): 36,75,276.37 (2024: ...)
                     loans_match = re.search(r'Loans \(net of ECL\):\s*₹?([\d,]+\.\d+)\s*lakhs', doc_content)
                     if loans_match:
                         axis_data['Loans (net of ECL) (Lakhs ₹)'] = float(loans_match.group(1).replace(',', ''))
                         logger.info(f"Extracted Axis Finance Loans (net of ECL): {axis_data.get('Loans (net of ECL) (Lakhs ₹)', 'Not found')} lakhs")


            # --- Extract Mahindra Finance data from deepseek_plaintext_20250604_4e8f84.txt content ---
            # Check if the document content seems to be for Mahindra Finance and contains relevant FY25/FY24 data
            if "Mahindra & Mahindra Financial Services Limited" in doc_content:
                 logger.info("Found potential Mahindra Finance data in context")

                 # Try to extract FY25 Revenue (from Key Financial Highlights Standalone section)
                 # Looking for format like: FY '25: ₹21,486.30 crore (vs...
                 fy25_revenue_match = re.search(r'Revenue from Operations:\s*.*?FY \'25:\s*₹?([\d,]+\.\d+)\s*crore', doc_content, re.DOTALL)
                 if fy25_revenue_match:
                      mahindra_data['Revenue from Operations (Crore ₹)'] = float(fy25_revenue_match.group(1).replace(',', ''))
                      logger.info(f"Extracted Mahindra Finance FY25 Revenue: {mahindra_data.get('Revenue from Operations (Crore ₹)', 'Not found')} crore")

                 # Try to extract FY25 Loan Book
                 # Looking for format like: Loan Book: Grew 17% to INR 119,673 crores
                 fy25_loan_book_match = re.search(r'Loan Book:\s*Grew.*?to INR\s*([\d,]+)\s*crores', doc_content)
                 if fy25_loan_book_match:
                      mahindra_data['Loan Book (Crore ₹)'] = float(fy25_loan_book_match.group(1).replace(',', ''))
                      logger.info(f"Extracted Mahindra Finance FY25 Loan Book: {mahindra_data.get('Loan Book (Crore ₹)', 'Not found')} crore")

                 # Extract FY24 PAT (Standalone Audited) from table
                 # Looking for format in table: | Profit After Tax (PAT) | ... | FY 2024 (Audited) | 1,673.61 |
                 fy24_pat_match = re.search(r'\| Profit After Tax \(PAT\)\s*\|.*?FY 2024 \(Audited\)\s*\|?\s*([\d,]+\.\d+)\s*\|', doc_content, re.DOTALL)
                 if fy24_pat_match:
                     mahindra_data['Profit After Tax (FY24 Standalone, Crore ₹)'] = float(fy24_pat_match.group(1).replace(',', ''))
                     logger.info(f"Extracted Mahindra Finance FY24 Standalone PAT: {mahindra_data.get('Profit After Tax (FY24 Standalone, Crore ₹)', 'Not found')} crore")


                 # Extract FY24 Total Assets (Standalone) from table
                 # Looking for format in table: | Total Assets | ... | 1,22,700.14 |
                 fy24_assets_match = re.search(r'Total Assets\s*\|.*?FY 2024 \(Audited\)\s*\|?\s*([\d,]+\.\d+)\s*\|', doc_content, re.DOTALL)
                 if fy24_assets_match:
                      mahindra_data['Total Assets (FY24 Standalone, Crore ₹)'] = float(fy24_assets_match.group(1).replace(',', ''))
                      logger.info(f"Extracted Mahindra Finance FY24 Standalone Assets: {mahindra_data.get('Total Assets (FY24 Standalone, Crore ₹)', 'Not found')} crore")


        logger.info(f"Raw Extracted Data - Axis Finance: {axis_data}, Mahindra Finance: {mahindra_data}")

        # --- Prepare data for DataFrame ----
        data_list = []

        # Add Axis Finance data if found
        if len(axis_data) > 1: # Check if metrics other than 'Company' were added
             logger.info("Adding Axis Finance data to list")
             if 'Revenue from Operations (Lakhs ₹)' in axis_data:
                 data_list.append({'Company': 'Axis Finance', 'Metric': 'Revenue (FY2025)', 'Amount (Crore ₹)': axis_data['Revenue from Operations (Lakhs ₹)'] / 100.0})
             if 'Profit After Tax (Lakhs ₹)' in axis_data:
                 data_list.append({'Company': 'Axis Finance', 'Metric': 'Profit After Tax (FY2025)', 'Amount (Crore ₹)': axis_data['Profit After Tax (Lakhs ₹)'] / 100.0})
             if 'Total Financial Assets (Lakhs ₹)' in axis_data:
                 data_list.append({'Company': 'Axis Finance', 'Metric': 'Total Financial Assets (FY2025)', 'Amount (Crore ₹)': axis_data['Total Financial Assets (Lakhs ₹)'] / 100.0})
             elif 'Loans (net of ECL) (Lakhs ₹)' in axis_data:
                 data_list.append({'Company': 'Axis Finance', 'Metric': 'Loans (net of ECL) (FY2025)', 'Amount (Crore ₹)': axis_data['Loans (net of ECL) (Lakhs ₹)'] / 100.0})

        # Add Mahindra Finance data if found
        if len(mahindra_data) > 1: # Check if metrics other than 'Company' were added
             logger.info("Adding Mahindra Finance data to list")
             if 'Revenue from Operations (Crore ₹)' in mahindra_data:
                 data_list.append({'Company': 'Mahindra Finance', 'Metric': 'Revenue (FY2025)', 'Amount (Crore ₹)': mahindra_data['Revenue from Operations (Crore ₹)']})
             if 'Profit After Tax (FY24 Standalone, Crore ₹)' in mahindra_data:
                 data_list.append({'Company': 'Mahindra Finance', 'Metric': 'Profit After Tax (FY2024 Standalone)', 'Amount (Crore ₹)': mahindra_data['Profit After Tax (FY24 Standalone, Crore ₹)']})
             if 'Loan Book (Crore ₹)' in mahindra_data:
                data_list.append({'Company': 'Mahindra Finance', 'Metric': 'Loan Book (FY2025)', 'Amount (Crore ₹)': mahindra_data['Loan Book (Crore ₹)']})
             elif 'Total Assets (FY24 Standalone, Crore ₹)' in mahindra_data:
                 data_list.append({'Company': 'Mahindra Finance', 'Metric': 'Total Assets (FY2024 Standalone)', 'Amount (Crore ₹)': mahindra_data['Total Assets (FY24 Standalone, Crore ₹)']})


        if not data_list:
            logger.warning("No relevant data extracted for graphing.")
            # Return an empty DataFrame if no data is found
            return pd.DataFrame()

        df = pd.DataFrame(data_list)
        logger.info(f"Final DataFrame constructed: {df.to_dict(orient='records')}")

        return df

    def generate_graph(self, query: str, context: List[str]) -> Dict[str, Any]:
        """Generate appropriate graph based on query and context."""
        try:
            logger.info(f"Generating graph for query: {query}")

            # Extract data from context
            df = self._extract_data_from_context(context, query)

            if df.empty:
                return {
                    "error": "Could not extract relevant financial data from the provided documents for comparison.",
                    "graph_data": None,
                    "embed_code": None,
                    "filepath": None
                }

            logger.info(f"Data extracted for plotting: {df.to_dict(orient='records')}")

            # Determine the type of graph based on extracted data and query
            # Use plotly express for grouped bar charts when comparing multiple metrics across companies
            if len(df['Metric'].unique()) > 1 and len(df['Company'].unique()) > 1:
                 fig = px.bar(df, x="Company", y="Amount (Crore ₹)", color="Metric",
                              title="Financial Performance Comparison", # Generic title
                              labels={'Amount (Crore ₹)': 'Amount (Crore ₹)'},
                              barmode='group',
                              text='Amount (Crore ₹)') # Display values on bars
                 fig.update_traces(texttemplate='%{text:.2s}', textposition='outside') # Format text on bars

            elif len(df['Company'].unique()) > 1: # Compare one metric across companies
                 metric_name = df['Metric'].iloc[0] if not df.empty else 'Value'
                 fig = go.Figure(data=[
                    go.Bar(
                        x=df['Company'],
                        y=df['Amount (Crore ₹)'],
                        text=df['Amount (Crore ₹)'].apply(lambda x: f'₹{x:,.2f} Cr'),
                        textposition='auto',
                    )
                 ])
                 fig.update_layout(
                    title=f'{metric_name} Comparison (Crore ₹)',
                    xaxis_title='Company',
                    yaxis_title='Amount (Crore ₹)',
                    template='plotly_white',
                    height=500,
                    width=800,
                    showlegend=False
                 )
            elif len(df['Metric'].unique()) > 1: # Show multiple metrics for one company
                 company_name = df['Company'].iloc[0] if not df.empty else 'Company'
                 fig = go.Figure(data=[
                    go.Bar(
                        x=df['Metric'],
                        y=df['Amount (Crore ₹)'],
                        text=df['Amount (Crore ₹)'].apply(lambda x: f'₹{x:,.2f} Cr'),
                        textposition='auto',
                    )
                 ])
                 fig.update_layout(
                    title=f'{company_name} Financial Metrics (Crore ₹)',
                    xaxis_title='Metric',
                    yaxis_title='Amount (Crore ₹)',
                    template='plotly_white',
                    height=500,
                    width=800,
                    showlegend=False
                 )

            else: # Fallback for single value
                 fig = go.Figure(data=[
                    go.Bar(
                        x=df['Company'],
                        y=df['Amount (Crore ₹)'],
                        text=df['Amount (Crore ₹)'].apply(lambda x: f'₹{x:,.2f} Cr'),
                        textposition='auto',
                    )
                 ])
                 fig.update_layout(
                    title='Financial Metric (Crore ₹)',
                    xaxis_title='Company',
                    yaxis_title='Amount (Crore ₹)',
                    template='plotly_white',
                    height=500,
                    width=800,
                    showlegend=False
                 )


            # Save the graph
            filepath, embed_code = self._save_graph(fig, "financial_comparison")
            logger.info("Graph generated and saved successfully")

            # Convert figure to JSON
            graph_json = json.loads(fig.to_json())

            return {
                "graph_type": "bar", # Assuming bar chart for comparison
                "graph_data": graph_json,
                "embed_code": embed_code,
                "filepath": filepath
            }

        except Exception as e:
            logger.error(f"Error generating graph: {str(e)}")
            return {
                "error": str(e),
                "graph_data": None,
                "embed_code": None,
                "filepath": None
            } 