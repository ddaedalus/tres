from models.preprocess_folds import *
from models.abcmodel import KwBiLSTM
from utils.hyperparameters import *
from configuration.config import *

import gc
import time
import tensorflow as tf

if __name__ == "__main__":

    # GPU configuration
    # -----------------
    if GPU_AVAILABLE:
        gpu = tf.config.experimental.list_physical_devices('GPU')

        if gpu:
            try:
                tf.config.experimental.set_virtual_device_configuration(gpu[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4500)])
            except RuntimeError as e:
                print(e)
        else:
            print("No GPU available")

    best_score = 0.0

    # 5-fold-cross-validation supervised learing, utilizing stratified batches
    # ------------------------------------------------------------------------
    best_scores_fold = []
    best_reports_fold = []
    for fold in range(FOLDS): 
        print(f"Fold {fold + 1}")
        train_ds = create_list_dataset(fold=fold, train=True)
        if OVERSAMPLING:
            train_ds = oversampling(train_ds)
        val_ds = create_list_dataset(fold=fold, train=False)

        t1 = time.time()

        if CLASSIFICATION_METHOD == "KwBiLSTM":
            model = KwBiLSTM(input_dim=300, maxseqlen=300, shortcut_dim1=SHORTCUT1, shortcut_dim2=3, output_dim=2, best_score=best_score)
        
        histories, best_report_fold = model.fit_ds(train_ds, val_ds, epochs=CLASSIFICATION_EPOCHS, batch_size=CLASSIFICATION_BATCH_SIZE)
        best_score = model.getBestScore()
        best_score_fold = model.getBestScoreFold()
        best_scores_fold.append( best_score_fold )
        best_reports_fold.append(best_report_fold)

        gc.collect()
        t2 = time.time()
        print(f"{(t2-t1)/60} minutes")
        print()
        print()

    print()
    print()
    print(f"Best mcc score: {best_score}")
    print()
    print()
    print(f"Best mcc scores per fold: {best_scores_fold}")
    print()
    print()
    for fold in range(FOLDS):
        print(f"Fold {fold + 1}")
        print(best_reports_fold[fold])
    print()
    print()
