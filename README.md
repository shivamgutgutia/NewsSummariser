# News Summariser

This project summarizes news articles using machine learning models. Follow the steps below to set up and run the project.

## Prerequisites

- Make sure you have Python installed, preferably version `3.11.5`. [Download Python](https://www.python.org/downloads/) if it's not installed.

## Setup

> **⚠️ Important Note:**  
> The first time you run this project, it may take a bit longer as base models are being downloaded and installed. **Please be patient.**

### Common Steps (For Both Model Training and Flask)

1. **Clone the repository**
    ```bash
    git clone https://github.com/shivamgutgutia/NewsSummariser.git
    ```
    ```bash
    cd NewsSummariser
    ```

2. **Create a virtual environment**
    Create a virtual environment named `venv` to manage project dependencies.
    ```bash
    python3 -m venv venv
    ```

3. **Activate the virtual environment**
    - On **Windows**:
        ```bash
        venv\Scripts\activate
        ```
    - On **MacOS/Linux**:
        ```bash
        source venv/bin/activate
        ```

4. **Install dependencies**
    Use the `requirements.txt` file to install all necessary Python modules.
    ```bash
    pip install -r requirements.txt
    ```

### Part 1: Model Training (Optional)

> **🔄 Optional Step**  
> A pre-trained model is already hosted on Hugging Face and can be used directly, so running this step is optional. However, if you'd like to train the model locally, follow these steps:

5. **Train the Model**
    Run the `modeltraining.py` script to train your model.
    ```bash
    python modeltraining.py
    ```

### Part 2: Running the Flask Application

6. **Start the Flask Application**
    Once the dependencies are installed, you can run the Flask app with the following command:
    ```bash
    flask run
    ```

7. **Open the Application**
    - Open your browser and go to [http://localhost:5000](http://localhost:5000) to access the application.

---

Enjoy using the News Summariser!
