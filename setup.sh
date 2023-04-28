#!/bin/bash

# Download model from gdown link
gdown 'https://drive.google.com/uc?id=1TxaMckkdOZ64XWs6RUBsDi5NSuZhBN1L' -O balaji.zip

# Unzip model file
unzip balaji.zip

# Create Streamlit config files
mkdir -p ~/.streamlit/
echo "\
[general]\n\
email = \"your-email@domain.com\"\n\
" > ~/.streamlit/credentials.toml
echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml
