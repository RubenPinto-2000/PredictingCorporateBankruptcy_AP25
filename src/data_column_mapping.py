"""
Column name mapping from the UCI Polish Bankruptcy dataset (id=365).
Maps generic feature labels (A1, A2, ...) to readable ratio names used in this project.
Naming follows the dataset documentation; some names are simplified for clarity.
"""

# NOTE: A9 ("Sales to total assets") is kept for traceability, but it will be excluded during cleaning.
# In accounting, "sales" and "total sales" are used interchangeably, making A9 highly redundant with A36
# ("Total sales to total assets"). Both map to "Asset turnover" in the short mapping.
# As the distinction intended by the dataset authors is not explicitly documented, I keep only A36, 
# which explicitly mentions "total sales" for correct interpretation, and exclude A9 to avoid 
# introducing potential multicollinearity without clear additional information.

# Original names from the dataset
COLUMN_MAPPING = {
    "year": "year",
    "A1": "Net profit to total assets",
    "A2": "Total liabilities to total assets",
    "A3": "Working capital to total assets",
    "A4": "Current assets to short term liabilities",
    "A5": "Cash flow to operating expenses",
    "A6": "Retained earnings to total assets",
    "A7": "EBIT to total assets",  # Earnings Before Interest and Taxes
    "A8": "Book value equity to total liabilities",
    "A9": "Sales to total assets",
    "A10": "Equity to total assets",
    "A11": "Gross profit extraordinary to total assets",
    "A12": "Gross profit to short term liabilities",
    "A13": "Gross profit depreciation to sales",
    "A14": "Gross profit interest to total assets",
    "A15": "Total liabilities days to gross profit depreciation",
    "A16": "Gross profit depreciation to total liabilities",
    "A17": "Total assets to total liabilities",
    "A18": "Gross profit to total assets",
    "A19": "Gross profit to sales",
    "A20": "Inventory days to sales",
    "A21": "Sales growth",
    "A22": "Profit operating activities to total assets",
    "A23": "Net profit to sales",
    "A24": "Gross profit 3years to total assets",
    "A25": "Equity minus share capital to total assets",
    "A26": "Net profit depreciation to total liabilities",
    "A27": "Profit operating activities to financial expenses",
    "A28": "Working capital to fixed assets",
    "A29": "Logarithm total assets",
    "A30": "Total liabilities minus cash to sales",
    "A31": "Gross profit interest to sales",
    "A32": "Current liabilities days to cost products sold",
    "A33": "Operating expenses to short term liabilities",
    "A34": "Operating expenses to total liabilities",
    "A35": "Profit sales to total assets",
    "A36": "Total sales to total assets",
    "A37": "Current assets minus inventory to long term liabilities",
    "A38": "Constant capital to total assets",
    "A39": "Profit sales to sales",
    "A40": "Current assets minus inventory minus receivables to short term liabilities",
    "A41": "Total liabilities to operating profit depreciation",
    "A42": "Profit operating activities to sales",
    "A43": "Rotation receivables plus inventory turnover days",
    "A44": "Receivables days to sales",
    "A45": "Net profit to inventory",
    "A46": "Current assets minus inventory to short term liabilities",
    "A47": "Inventory days to cost products sold",
    "A48": "EBITDA to total assets",  # Earnings Before Interest, Taxes, Depreciation and Amortization
    "A49": "EBITDA to sales",
    "A50": "Current assets to total liabilities",
    "A51": "Short term liabilities to total assets",
    "A52": "Short term liabilities days to cost products sold",
    "A53": "Equity to fixed assets",
    "A54": "Constant capital to fixed assets",
    "A55": "Working capital",
    "A56": "Sales minus cost to sales",
    "A57": "Current assets minus inventory minus short term liabilities to sales minus gross profit depreciation",
    "A58": "Total costs to total sales",
    "A59": "Long term liabilities to equity",
    "A60": "Sales to inventory",
    "A61": "Sales to receivables",
    "A62": "Short term liabilities days to sales",
    "A63": "Sales to short term liabilities",
    "A64": "Sales to fixed assets",
    "class": "Bankruptcy status"
}

# A shorter and more readable names for easier interpretation and plots
COLUMN_MAPPING_SHORT = {
    "year": "year",
    "A1": "ROA",  # Return on Assets
    "A2": "Debt ratio",
    "A3": "Working capital ratio",
    "A4": "Current ratio",
    "A5": "Cash flow coverage",
    "A6": "Retained earnings ratio",
    "A7": "EBIT ROA",  # Earnings Before Interest and Taxes
    "A8": "Equity to debt",
    "A9": "Asset turnover",  # FLAGGED with A36
    "A10": "Equity ratio",
    "A11": "Gross profit extra ratio",
    "A12": "Gross profit to current debt",
    "A13": "Gross margin depreciation",
    "A14": "Gross profit interest ROA",
    "A15": "Debt payback days",
    "A16": "Gross profit depreciation to debt",
    "A17": "Asset to debt",
    "A18": "Gross profit ROA",
    "A19": "Gross margin rate",
    "A20": "Inventory turnover days",
    "A21": "Sales growth",
    "A22": "Operating profit ROA",
    "A23": "Net margin",
    "A24": "Gross profit 3y ROA",
    "A25": "Equity minus capital ratio",
    "A26": "Net profit depreciation to debt",
    "A27": "Operating profit to financial cost",
    "A28": "Working capital to fixed",
    "A29": "Log assets",
    "A30": "Debt minus cash to sales",
    "A31": "Gross profit interest margin",
    "A32": "Current debt days to COGS",  # Cost of Goods Sold
    "A33": "Operating expenses to current debt",
    "A34": "Operating expenses to debt",
    "A35": "Profit sales ROA",
    "A36": "Asset turnover",  # FLAGGED with A36
    "A37": "Quick assets to long term debt",
    "A38": "Constant capital ratio",
    "A39": "Profit margin",
    "A40": "Strict liquidity ratio",
    "A41": "Debt to operating profit depreciation",
    "A42": "Operating margin",
    "A43": "Receivables inventory days",
    "A44": "Receivables turnover days",
    "A45": "Net profit to inventory",
    "A46": "Quick ratio",
    "A47": "Inventory turnover days COGS",
    "A48": "EBITDA ROA",  # Earnings Before Interest, Taxes, Depreciation and Amortization
    "A49": "EBITDA margin",
    "A50": "Current assets to debt",
    "A51": "Current debt ratio",
    "A52": "Current debt days COGS",
    "A53": "Equity to fixed",
    "A54": "Constant capital to fixed",
    "A55": "Working capital abs",  #Absolute
    "A56": "Gross margin",
    "A57": "Liquidity to adjusted sales",
    "A58": "Cost ratio",
    "A59": "Long-term debt to equity",
    "A60": "Inventory turnover",
    "A61": "Receivables turnover",
    "A62": "Current debt days sales",
    "A63": "Sales to current debt",
    "A64": "Fixed asset turnover",
    "class": "Bankruptcy status"
}


def detect_duplicate_mappings(mapping_dict):
    """
    Identify duplicated target names in a column-mapping dictionary.
    This allows subsequent removal of duplicate mappings to avoid multicollinearity.
    """
    value_to_keys = {}
    for key, value in mapping_dict.items():
        # Skip keys that are not financial ratios
        if (key in ["year", "class"]):
            continue
        if (value not in value_to_keys):
            value_to_keys[value] = []
        value_to_keys[value].append(key)
    
    # Return only duplicates 
    duplicates = {value: keys for value, keys in value_to_keys.items() if len(keys) > 1}
    return duplicates


def rename_columns(dataframe):
    """
    Rename dataset columns using the project mapping short names.
    This makes plots more readable and clearer by using descriptive names instead of formula names
    """
    columns_to_rename = {old: new for old, new in COLUMN_MAPPING_SHORT.items() if old in dataframe.columns}
    dataframe_renamed = dataframe.rename(columns=columns_to_rename)
    
    return dataframe_renamed
