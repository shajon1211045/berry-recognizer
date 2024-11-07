# Berry Type Recognizer
This project utilizes **fastai** and **ResNet50** to build an image recognition model that achieves high accuracy in distinguishing various types of berries. The model is evaluated using **error rate** and **accuracy** and achieves an accuracy of approximately **~98%**. The training dataset contains a diverse collection of berry images to enhance model performance.

## Goal of the Project
The goal of this project is to accurately recognize different types of berries. This application could assist in food industry applications or educational tools for berry identification. The types of berries included are as follows:
1. Strawberry
2. Raspberry
3. Blueberry
4. Blackberry
5. Cranberry

## Dataset Preparation
**Data Collection:** Images of different berry types were collected from various sources online using the berry names as labels.<br/>
**DataLoader:** The data loader was set up using the fastai `DataBlock` API to manage the dataset efficiently.<br/>
**Data Augmentation:** Fastai's built-in data augmentation capabilities were used to ensure variability, and the details can be found in the `notebooks/data_preparation.ipynb` file.<br/>

## Model Training and Data Cleaning
**Model Training:** The model was trained using **ResNet50** with transfer learning. The training process included multiple stages, starting with 10 epochs and gradually fine-tuning with 5, 3, and 2 epochs, resulting in a final accuracy of approximately **~98%** and an error rate below **2%**.

**Data Cleaning:** Cleaning the dataset based on the confusion matrix scores was essential to improving the model's accuracy. Using fastai's `ImageClassifierCleaner`, the dataset was iteratively refined to remove misclassified or irrelevant images, significantly improving the model's performance.

## Model Deployment
The final trained model (`berry_recogniser_v2.pkl`) is deployed on Hugging Face Spaces using Gradio for user interaction. The deployment code is available in the `deployment` folder, or you can try it [here](https://huggingface.co/spaces/shajon/berry-recogniser).

![Hugging Face Space Deployment](https://via.placeholder.com/858x300)

## API Integration with Website
A simple, two-page website was created using **HTML, CSS, and JavaScript**, where the model's API is integrated for interactive berry recognition. The site is hosted using GitHub Pages, and the code for the site can be found in the `docs` folder.

![Index Page](https://via.placeholder.com/300x150)
![Output Page](https://via.placeholder.com/300x150)

## Get Started

`Python version 3.10 or 3.11 is required.`

To set up and run the project locally, follow these steps:

1. Clone the project repository from GitHub:

    ```bash
    git clone https://github.com/shajon1211045/berry-recogniser.git
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Prepare the berry dataset by running the data preparation notebook: [data_preparation.ipynb](notebooks/data_preparation.ipynb).

4. Train the model using the dataset by running: [modelTraining_and_dataCleaning.ipynb](notebooks/modelTraining_and_dataCleaning.ipynb).

5. To test the model locally, navigate to the deployment folder and run the app:

    ```bash
    cd deployment
    python app.py
    ```

Alternatively, you can access the model directly on [Hugging Face Space: shajon/berry-recogniser](https://huggingface.co/spaces/shajon/berry-recogniser) to recognize different berry types.

---

For any questions, feel free to reach out via LinkedIn or GitHub: 
- LinkedIn: [Md Zahidul Islam](https://www.linkedin.com/in/zahidulshajon/)
- GitHub: [shajon1211045](https://github.com/shajon1211045)

