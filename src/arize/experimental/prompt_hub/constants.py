"""
Constants for the Arize Prompt Hub.
"""

# Mapping from internal model names to external model names
ARIZE_INTERNAL_MODEL_MAPPING = {
    # ----------------------
    #   OpenAI Models
    # ----------------------
    "GPT_4o_MINI": "gpt-4o-mini",
    "GPT_4o_MINI_2024_07_18": "gpt-4o-mini-2024-07-18",
    "GPT_4o": "gpt-4o",
    "GPT_4o_2024_05_13": "gpt-4o-2024-05-13",
    "GPT_4o_2024_08_06": "gpt-4o-2024-08-06",
    "CHATGPT_4o_LATEST": "chatgpt-4o-latest",
    "O1_PREVIEW": "o1-preview",
    "O1_PREVIEW_2024_09_12": "o1-preview-2024-09-12",
    "O1_MINI": "o1-mini",
    "O1_MINI_2024_09_12": "o1-mini-2024-09-12",
    "GPT_4_TURBO": "gpt-4-turbo",
    "GPT_4_TURBO_2024_04_09": "gpt-4-turbo-2024-04-09",
    "GPT_4_TURBO_PREVIEW": "gpt-4-turbo-preview",
    "GPT_4_0125_PREVIEW": "gpt-4-0125-preview",
    "GPT_4_1106_PREVIEW": "gpt-4-1106-preview",
    "GPT_4": "gpt-4",
    "GPT_4_32k": "gpt-4-32k",
    "GPT_4_0613": "gpt-4-0613",
    "GPT_4_0314": "gpt-4-0314",
    "GPT_4_VISION_PREVIEW": "gpt-4-vision-preview",
    "GPT_3_5_TURBO": "gpt-3.5-turbo",
    "GPT_3_5_TURBO_1106": "gpt-3.5-turbo-1106",
    "GPT_3_5_TURBO_INSTRUCT": "gpt-3.5-turbo-instruct",
    "GPT_3_5_TURBO_0125": "gpt-3.5-turbo-0125",
    "TEXT_DAVINCI_003": "text-davinci-003",
    "BABBAGE_002": "babbage-002",
    "DAVINCI_002": "davinci-002",
    "O1_2024_12_17": "o1-2024-12-17",
    "O1": "o1",
    "O3_MINI": "o3-mini",
    "O3_MINI_2025_01_31": "o3-mini-2025-01-31",
    # ----------------------
    #   Vertex AI Models
    # ----------------------
    "GEMINI_1_0_PRO": "gemini-1.0-pro",
    "GEMINI_1_0_PRO_VISION": "gemini-1.0-pro-vision",
    "GEMINI_1_5_FLASH": "gemini-1.5-flash",
    "GEMINI_1_5_FLASH_002": "gemini-1.5-flash-002",
    "GEMINI_1_5_FLASH_8B": "gemini-1.5-flash-8b",
    "GEMINI_1_5_FLASH_LATEST": "gemini-1.5-flash-latest",
    "GEMINI_1_5_FLASH_8B_LATEST": "gemini-1.5-flash-8b-latest",
    "GEMINI_1_5_PRO": "gemini-1.5-pro",
    "GEMINI_1_5_PRO_001": "gemini-1.5-pro-001",
    "GEMINI_1_5_PRO_002": "gemini-1.5-pro-002",
    "GEMINI_1_5_PRO_LATEST": "gemini-1.5-pro-latest",
    "GEMINI_2_0_FLASH": "gemini-2.0-flash",
    "GEMINI_2_0_FLASH_001": "gemini-2.0-flash-001",
    "GEMINI_2_0_FLASH_EXP": "gemini-2.0-flash-exp",
    "GEMINI_2_0_FLASH_LITE_PREVIEW_02_05": "gemini-2.0-flash-lite-preview-02-05",
    "GEMINI_PRO": "gemini-pro",
}

# Reverse mapping from external model names to internal model names
ARIZE_EXTERNAL_MODEL_MAPPING = {
    v: k for k, v in ARIZE_INTERNAL_MODEL_MAPPING.items()
}
