""" WIP """
""" Current code based on Process_02_v01_Inference_on_test_ResNet50.ipynb"""

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import fastai
from fastai.vision.all import *
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

# Custom imports
from src.utils import plot_class_distribution, plot_confusion_matrix, plot_classification_metrics

def evaluate_model(MODEL_NAME, BATCH_SIZE, images_path, REFERENCE_PATH, TRAIN_DATASET):
    np.random.seed(3)

    # Get image paths
    block = DataBlock(blocks=(ImageBlock, CategoryBlock), # for regression, change this CategoryBlock
                    splitter=RandomSplitter(),
                    get_items = get_image_files, 
                    get_y = parent_label,
                    #item_tfms=Resize((770,1040), method='squish'), 
                    item_tfms=Resize(300, ResizeMethod.Pad, pad_mode='zeros'), # see page 73 book
                    batch_tfms=[*aug_transforms( #https://docs.fast.ai/vision.augment#aug_transforms
                        mult=1.0, 
                        do_flip=True,
                        flip_vert=True,
                        max_rotate=0.2,
                        min_zoom=1.0,
                        max_zoom=1.1,
                        max_lighting=0.3,
                        max_warp=0.1,
                        p_affine=0.5,
                        p_lighting=0.5,
                        pad_mode='zeros'), 
                                Normalize.from_stats(*imagenet_stats)] 
                    )

    dls = block.dataloaders(TRAIN_DATASET, bs=BATCH_SIZE)


    # Create Learner
    learn = vision_learner(dls, resnet50, #loss_func=Huber(), # different ResNets and Loss functions can be specified here
                        metrics=error_rate
                    ); # creates pretrained model
    learn.model = torch.nn.DataParallel(learn.model) # Parallels computations over multiplle GPUs

    # Get images to predict
    imgs = get_image_files(images_path)

    print(f'[INFO] This is Plankton Identifier version: {MODEL_NAME}')
    print(f'[INFO] The batchsize is set at: {BATCH_SIZE}')
    print(f'[INFO] The loss function is: {learn.loss_func}') # Double check current loss func
    print(f'[INFO] {len(imgs):,} images in {images_path}')


    learn.load(MODEL_NAME, weights_only=False)

    print(f'The path to the first image is: {imgs[0]}')
    print(f'The path to the last image is: {imgs[-1]}')

    # Create image batch for predicting
    dl = learn.dls.test_dl(imgs)

    # Get predictions for image batch (for large dataset (10k images), this takes ~45min)
    preds, _, label_numeric = learn.get_preds(dl=dl, with_decoded=True)
    print(f"[INFO] Made predictions")

    # Create table (i.e. DataFrame) with predictions
    test_df = pd.DataFrame()  # Creates empty table
    test_df['id'] = imgs  # Adds full filepaths
    test_df['filename'] = test_df['id'].apply(lambda x: os.path.basename(x)) # Extract filenames from the full paths in test_df
    test_df['pred_id'] = label_numeric  # Adds predictions to table

    # Create new column for predictions labels
    label_mapping = {i: class_name for i, class_name in enumerate(learn.dls.vocab)}
    test_df['pred_label'] = test_df['pred_id'].map(label_mapping)

    # Load the ground truth CSV file
    ground_truth_df = pd.read_csv(REFERENCE_PATH)

    # Rename 'Subdirectory' column to 'Label'
    # TODO: Fix the entire label CSV; either automated from file structure
    ground_truth_df.rename(columns={'Subdirectory': 'label', 'Filename':'filename'}, inplace=True)

    # Merge test_df and ground_truth_df on 'Filename'
    merged_df = pd.merge(test_df, ground_truth_df, on='filename', how='inner')

    # Export table to .csv file for downstream applications
    csv_path = os.path.join("data", f"{MODEL_NAME}_evaluate.csv")
    merged_df.to_csv(csv_path, index=False, float_format='%.9f')

    # Extracting the true labels and predictions
    y_true = merged_df['label']
    y_pred = merged_df['pred_label']

    # Calculate and display metrics
    correct_predictions = y_pred == y_true
    num_correct_predictions = correct_predictions.sum()

    # Calculating precision, recall and F1 with 'macro' average
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro')

    # Printing the results
    print(f"Total number of images in test set:\t {len(merged_df):,}")
    print(f"Total number of Correct Predictions:\t {num_correct_predictions:,}")
    print(f"Precision (macro-average):\t\t {precision * 100:.2f}%")
    print(f"Recall (macro-average):\t\t\t {recall * 100:.2f}%")
    print(f"F1 Score (macro-average):\t\t {f1 * 100:.2f}%")

    # Ensure the 'Label' column is sorted alphabetically
    sorted_labels = merged_df['label'].sort_values()

    # Create images root
    images_root = f"doc/{datetime.today().strftime('%Y-%m-%d')}_{MODEL_NAME}_eval"
    os.makedirs(images_root, exist_ok=True)

    # Call visualization functions
    plot_class_distribution(images_root, sorted_labels)
    plot_confusion_matrix(images_root, MODEL_NAME, merged_df, y_true, y_pred)
    plot_classification_metrics(images_root, merged_df)

    print(F"[INFO] Finished evaluate.py")

# Hard-coded variables
MODEL_NAME = 'ResNet50PlanktoFAIR_WeightedLoss' # Insert your model filename; see train.py
BATCH_SIZE = 128
REFERENCE_PATH = 'data/DETAILED_test.csv'
TRAIN_DATASET = Path('data/DETAILED_merged')
TEST_DATASET = Path('data/DETAILED_test')

# Execute function
evaluate_model(MODEL_NAME, BATCH_SIZE, TEST_DATASET, REFERENCE_PATH, TRAIN_DATASET)