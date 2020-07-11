mkdir -p ~/.streamlit/
echo "\
[general]\n\
email = \"sauhaarda@mit.edu\"\n\
" > ~/.streamlit/credentials.toml
echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
