# YouTube Assistant

Ask questions about any YouTube video to this LLM powered assistant.

NOTE: YouTube transcript loader sometimes fails and return an empty object. It is not an issue with GenAI, but with the data load. When This happens the program stops.

## Running it locally

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the streamlit app:

```bash
streamlit run main.py
```

![YouTube Assistant App](/YouTube-Assistant.png)

## Hosted On

The web-app uses streamlit and is hosted on [Azure Container Apps.](https://azure.microsoft.com/en-ca/products/container-apps)

## Author

- LinkedIn: [Andrianov Alexander](https://www.linkedin.com/in/alexander--andrianov)