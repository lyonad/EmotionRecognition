import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os
import pickle
import time
from datetime import datetime
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                           confusion_matrix, classification_report, roc_curve, auc,
                           precision_recall_curve, average_precision_score)
from sklearn.model_selection import cross_val_score, StratifiedKFold, learning_curve
from sklearn.preprocessing import label_binarize
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    """
    Kelas untuk evaluasi komprehensif model deteksi emosi
    """
    def __init__(self, model_path='emotion_model_v1.pkl'):
        self.model_path = model_path
        self.knn = None
        self.pca = None
        self.scaler = None
        self.emotions = None
        self.results = {}
        
    def load_model(self):
        """
        Memuat model dari file
        """
        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            self.knn = model_data['knn']
            self.pca = model_data['pca']
            self.scaler = model_data['scaler']
            self.emotions = model_data['emotions']
            print(f"Model berhasil dimuat dari {self.model_path}")
            return True
        except Exception as e:
            print(f"Error memuat model: {e}")
            return False
    
    def load_dataset(self, test_dir):
        """
        Memuat dataset untuk evaluasi
        """
        X_test = []
        y_test = []
        
        print(f"Memuat dataset dari: {test_dir}")
        
        for emotion_idx, emotion in enumerate(self.emotions):
            emotion_dir = os.path.join(test_dir, emotion)
            if os.path.isdir(emotion_dir):
                files = [f for f in os.listdir(emotion_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
                
                for image_file in files:
                    img_path = os.path.join(emotion_dir, image_file)
                    try:
                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            img = cv2.resize(img, (48, 48))
                            X_test.append(img.flatten())
                            y_test.append(emotion_idx)
                    except Exception as e:
                        print(f"Error loading {img_path}: {e}")
        
        print(f"Total sampel test: {len(X_test)}")
        return np.array(X_test), np.array(y_test)
    
    def evaluate_basic_metrics(self, X_test, y_test):
        """
        Evaluasi metrik dasar (accuracy, precision, recall, f1-score)
        """
        print("\n=== Evaluasi Metrik Dasar ===")
        
        # Preprocess data
        X_test_scaled = self.scaler.transform(X_test)
        X_test_pca = self.pca.transform(X_test_scaled)
        
        # Prediksi
        start_time = time.time()
        y_pred = self.knn.predict(X_test_pca)
        pred_time = time.time() - start_time
        
        # Hitung metrik
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Simpan hasil
        self.results['basic_metrics'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'prediction_time': pred_time,
            'avg_prediction_time_per_sample': pred_time / len(y_test)
        }
        
        # Print hasil
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision (weighted): {precision:.4f}")
        print(f"Recall (weighted): {recall:.4f}")
        print(f"F1-Score (weighted): {f1:.4f}")
        print(f"Waktu prediksi total: {pred_time:.4f} detik")
        print(f"Waktu prediksi per sampel: {pred_time/len(y_test)*1000:.2f} ms")
        
        # Per-class metrics
        print("\n=== Metrik Per Kelas ===")
        precision_per_class = precision_score(y_test, y_pred, average=None)
        recall_per_class = recall_score(y_test, y_pred, average=None)
        f1_per_class = f1_score(y_test, y_pred, average=None)
        
        metrics_df = pd.DataFrame({
            'Emotion': self.emotions,
            'Precision': precision_per_class,
            'Recall': recall_per_class,
            'F1-Score': f1_per_class
        })
        print(metrics_df.to_string(index=False))
        
        self.results['per_class_metrics'] = metrics_df
        self.results['y_test'] = y_test
        self.results['y_pred'] = y_pred
        
        return y_pred
    
    def plot_confusion_matrix(self, y_test, y_pred, save_path='evaluation_results/'):
        """
        Plot confusion matrix dengan detail
        """
        print("\n=== Confusion Matrix ===")
        
        # Buat direktori jika belum ada
        os.makedirs(save_path, exist_ok=True)
        
        # Hitung confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Normalisasi confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot confusion matrix biasa
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.emotions, yticklabels=self.emotions)
        plt.title('Confusion Matrix', fontsize=16)
        plt.xlabel('Predicted', fontsize=14)
        plt.ylabel('True', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'confusion_matrix.png'), dpi=300)
        plt.close()
        
        # Plot normalized confusion matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='YlOrRd',
                    xticklabels=self.emotions, yticklabels=self.emotions)
        plt.title('Normalized Confusion Matrix', fontsize=16)
        plt.xlabel('Predicted', fontsize=14)
        plt.ylabel('True', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'confusion_matrix_normalized.png'), dpi=300)
        plt.close()
        
        # Analisis kesalahan klasifikasi
        print("\nAnalisis Kesalahan Klasifikasi:")
        for i, emotion in enumerate(self.emotions):
            misclassified = []
            for j, pred_emotion in enumerate(self.emotions):
                if i != j and cm[i, j] > 0:
                    misclassified.append(f"{pred_emotion} ({cm[i, j]})")
            if misclassified:
                print(f"{emotion} sering salah diklasifikasi sebagai: {', '.join(misclassified)}")
        
        self.results['confusion_matrix'] = cm
        self.results['confusion_matrix_normalized'] = cm_normalized
    
    def plot_roc_curves(self, X_test, y_test, save_path='evaluation_results/'):
        """
        Plot ROC curves untuk setiap kelas
        """
        print("\n=== ROC Curves ===")
        
        # Preprocess data
        X_test_scaled = self.scaler.transform(X_test)
        X_test_pca = self.pca.transform(X_test_scaled)
        
        # Binarize labels untuk multi-class ROC
        y_test_bin = label_binarize(y_test, classes=range(len(self.emotions)))
        
        # Dapatkan probabilitas prediksi
        y_score = self.knn.predict_proba(X_test_pca)
        
        # Plot ROC untuk setiap kelas
        plt.figure(figsize=(12, 10))
        
        roc_auc = {}
        for i in range(len(self.emotions)):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{self.emotions[i]} (AUC = {roc_auc[i]:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=14)
        plt.ylabel('True Positive Rate', fontsize=14)
        plt.title('ROC Curves untuk Setiap Emosi', fontsize=16)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'roc_curves.png'), dpi=300)
        plt.close()
        
        # Print AUC scores
        print("\nArea Under Curve (AUC) per kelas:")
        auc_df = pd.DataFrame({
            'Emotion': self.emotions,
            'AUC': [roc_auc[i] for i in range(len(self.emotions))]
        })
        print(auc_df.to_string(index=False))
        
        self.results['roc_auc'] = roc_auc
    
    def plot_precision_recall_curves(self, X_test, y_test, save_path='evaluation_results/'):
        """
        Plot Precision-Recall curves
        """
        print("\n=== Precision-Recall Curves ===")
        
        # Preprocess data
        X_test_scaled = self.scaler.transform(X_test)
        X_test_pca = self.pca.transform(X_test_scaled)
        
        # Binarize labels
        y_test_bin = label_binarize(y_test, classes=range(len(self.emotions)))
        
        # Dapatkan probabilitas prediksi
        y_score = self.knn.predict_proba(X_test_pca)
        
        # Plot PR curves
        plt.figure(figsize=(12, 10))
        
        avg_precision = {}
        for i in range(len(self.emotions)):
            precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
            avg_precision[i] = average_precision_score(y_test_bin[:, i], y_score[:, i])
            plt.plot(recall, precision, label=f'{self.emotions[i]} (AP = {avg_precision[i]:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=14)
        plt.ylabel('Precision', fontsize=14)
        plt.title('Precision-Recall Curves untuk Setiap Emosi', fontsize=16)
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'precision_recall_curves.png'), dpi=300)
        plt.close()
        
        self.results['avg_precision'] = avg_precision
    
    def analyze_feature_importance(self, X_test, y_test, save_path='evaluation_results/'):
        """
        Analisis pentingnya fitur menggunakan PCA components
        """
        print("\n=== Analisis Feature Importance ===")
        
        # Variance explained by PCA components
        variance_ratio = self.pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(variance_ratio)
        
        # Plot variance explained
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.bar(range(1, len(variance_ratio) + 1), variance_ratio)
        plt.xlabel('Principal Component', fontsize=12)
        plt.ylabel('Variance Ratio', fontsize=12)
        plt.title('Variance Explained by Each PC', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'bo-')
        plt.axhline(y=0.95, color='r', linestyle='--', label='95% variance')
        plt.xlabel('Number of Components', fontsize=12)
        plt.ylabel('Cumulative Variance Ratio', fontsize=12)
        plt.title('Cumulative Variance Explained', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'pca_variance_analysis.png'), dpi=300)
        plt.close()
        
        # Find number of components for 95% variance
        n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
        print(f"Jumlah komponen untuk 95% variance: {n_components_95}")
        print(f"Total variance explained: {cumulative_variance[-1]:.4f}")
        
        self.results['pca_analysis'] = {
            'variance_ratio': variance_ratio,
            'cumulative_variance': cumulative_variance,
            'n_components_95': n_components_95
        }
    
    def cross_validation_analysis(self, X_test, y_test, save_path='evaluation_results/'):
        """
        Analisis cross-validation
        """
        print("\n=== Cross-Validation Analysis ===")
        
        # Preprocess data
        X_test_scaled = self.scaler.transform(X_test)
        X_test_pca = self.pca.transform(X_test_scaled)
        
        # Stratified K-Fold
        cv_folds = 5
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Cross-validation scores
        cv_scores = cross_val_score(self.knn, X_test_pca, y_test, cv=skf, scoring='accuracy')
        
        print(f"Cross-Validation Scores ({cv_folds}-fold):")
        for i, score in enumerate(cv_scores):
            print(f"  Fold {i+1}: {score:.4f}")
        print(f"Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Plot CV scores
        plt.figure(figsize=(10, 6))
        plt.bar(range(1, cv_folds + 1), cv_scores, color='skyblue', edgecolor='navy')
        plt.axhline(y=cv_scores.mean(), color='r', linestyle='--', 
                    label=f'Mean: {cv_scores.mean():.4f}')
        plt.xlabel('Fold', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title(f'{cv_folds}-Fold Cross-Validation Results', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'cross_validation_scores.png'), dpi=300)
        plt.close()
        
        self.results['cv_scores'] = cv_scores
    
    def analyze_prediction_confidence(self, X_test, y_test, save_path='evaluation_results/'):
        """
        Analisis confidence level dari prediksi
        """
        print("\n=== Analisis Prediction Confidence ===")
        
        # Preprocess data
        X_test_scaled = self.scaler.transform(X_test)
        X_test_pca = self.pca.transform(X_test_scaled)
        
        # Dapatkan probabilitas prediksi
        y_proba = self.knn.predict_proba(X_test_pca)
        y_pred = self.knn.predict(X_test_pca)
        
        # Confidence scores (max probability)
        confidence_scores = np.max(y_proba, axis=1)
        
        # Analisis berdasarkan benar/salah
        correct_mask = y_pred == y_test
        correct_confidence = confidence_scores[correct_mask]
        incorrect_confidence = confidence_scores[~correct_mask]
        
        # Plot distribusi confidence
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.hist(correct_confidence, bins=30, alpha=0.7, label='Correct Predictions', color='green')
        plt.hist(incorrect_confidence, bins=30, alpha=0.7, label='Incorrect Predictions', color='red')
        plt.xlabel('Confidence Score', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Distribution of Confidence Scores', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        # Confidence by emotion
        confidence_by_emotion = []
        for i in range(len(self.emotions)):
            mask = y_test == i
            confidence_by_emotion.append(confidence_scores[mask])
        
        plt.boxplot(confidence_by_emotion, labels=self.emotions)
        plt.xlabel('Emotion', fontsize=12)
        plt.ylabel('Confidence Score', fontsize=12)
        plt.title('Confidence Scores by Emotion', fontsize=14)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'confidence_analysis.png'), dpi=300)
        plt.close()
        
        print(f"Mean confidence (correct): {correct_confidence.mean():.4f}")
        print(f"Mean confidence (incorrect): {incorrect_confidence.mean():.4f}")
        
        # Confidence threshold analysis
        thresholds = np.arange(0.3, 1.0, 0.05)
        accuracies = []
        coverages = []
        
        for threshold in thresholds:
            high_conf_mask = confidence_scores >= threshold
            if np.sum(high_conf_mask) > 0:
                high_conf_accuracy = np.sum(y_pred[high_conf_mask] == y_test[high_conf_mask]) / np.sum(high_conf_mask)
                coverage = np.sum(high_conf_mask) / len(y_test)
            else:
                high_conf_accuracy = 0
                coverage = 0
            accuracies.append(high_conf_accuracy)
            coverages.append(coverage)
        
        # Plot threshold analysis
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, accuracies, 'b-', label='Accuracy')
        plt.plot(thresholds, coverages, 'r-', label='Coverage')
        plt.xlabel('Confidence Threshold', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.title('Accuracy vs Coverage at Different Confidence Thresholds', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'confidence_threshold_analysis.png'), dpi=300)
        plt.close()
        
        self.results['confidence_analysis'] = {
            'mean_correct_confidence': correct_confidence.mean(),
            'mean_incorrect_confidence': incorrect_confidence.mean(),
            'confidence_scores': confidence_scores
        }
    
    def generate_report(self, save_path='evaluation_results/'):
        """
        Generate comprehensive evaluation report
        """
        print("\n=== Generating Evaluation Report ===")
        
        report_path = os.path.join(save_path, 'evaluation_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("EMOTION RECOGNITION MODEL EVALUATION REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model Path: {self.model_path}\n\n")
            
            # Basic Metrics
            if 'basic_metrics' in self.results:
                f.write("BASIC METRICS\n")
                f.write("-" * 40 + "\n")
                metrics = self.results['basic_metrics']
                f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
                f.write(f"Precision (weighted): {metrics['precision']:.4f}\n")
                f.write(f"Recall (weighted): {metrics['recall']:.4f}\n")
                f.write(f"F1-Score (weighted): {metrics['f1_score']:.4f}\n")
                f.write(f"Prediction Time per Sample: {metrics['avg_prediction_time_per_sample']*1000:.2f} ms\n\n")
            
            # Per-class Metrics
            if 'per_class_metrics' in self.results:
                f.write("PER-CLASS METRICS\n")
                f.write("-" * 40 + "\n")
                f.write(self.results['per_class_metrics'].to_string(index=False))
                f.write("\n\n")
            
            # Cross-validation
            if 'cv_scores' in self.results:
                f.write("CROSS-VALIDATION RESULTS\n")
                f.write("-" * 40 + "\n")
                cv_scores = self.results['cv_scores']
                f.write(f"Mean CV Score: {cv_scores.mean():.4f}\n")
                f.write(f"Std Dev: {cv_scores.std():.4f}\n\n")
            
            # Confidence Analysis
            if 'confidence_analysis' in self.results:
                f.write("CONFIDENCE ANALYSIS\n")
                f.write("-" * 40 + "\n")
                conf = self.results['confidence_analysis']
                f.write(f"Mean Confidence (Correct): {conf['mean_correct_confidence']:.4f}\n")
                f.write(f"Mean Confidence (Incorrect): {conf['mean_incorrect_confidence']:.4f}\n\n")
            
            # PCA Analysis
            if 'pca_analysis' in self.results:
                f.write("PCA ANALYSIS\n")
                f.write("-" * 40 + "\n")
                pca = self.results['pca_analysis']
                f.write(f"Components for 95% variance: {pca['n_components_95']}\n")
                f.write(f"Total variance explained: {pca['cumulative_variance'][-1]:.4f}\n\n")
            
            # Classification Report
            if 'y_test' in self.results and 'y_pred' in self.results:
                f.write("DETAILED CLASSIFICATION REPORT\n")
                f.write("-" * 40 + "\n")
                f.write(classification_report(self.results['y_test'], 
                                            self.results['y_pred'], 
                                            target_names=self.emotions))
        
        print(f"Report saved to: {report_path}")
        
        # Save results as pickle for later use
        results_pkl = os.path.join(save_path, 'evaluation_results.pkl')
        with open(results_pkl, 'wb') as f:
            pickle.dump(self.results, f)
        print(f"Results saved to: {results_pkl}")

def main():
    """
    Main function untuk menjalankan evaluasi
    """
    print("=== EMOTION RECOGNITION MODEL EVALUATION ===\n")
    
    # Path settings
    model_path = input("Masukkan path model (default: emotion_model_v1.pkl): ").strip()
    if not model_path:
        model_path = 'emotion_model_v1.pkl'
    
    test_dir = input("Masukkan path direktori test (default: C:\\Users\\LyonA\\Downloads\\archive\\test): ").strip()
    if not test_dir:
        test_dir = r'C:\Users\LyonA\Downloads\archive\test'
    
    save_path = input("Masukkan path untuk menyimpan hasil (default: evaluation_results/): ").strip()
    if not save_path:
        save_path = 'evaluation_results/'
    
    # Create evaluator
    evaluator = ModelEvaluator(model_path)
    
    # Load model
    if not evaluator.load_model():
        print("Gagal memuat model. Program berhenti.")
        return
    
    # Load test data
    X_test, y_test = evaluator.load_dataset(test_dir)
    
    if len(X_test) == 0:
        print("Dataset kosong. Program berhenti.")
        return
    
    print(f"\nDataset loaded: {len(X_test)} samples")
    
    # Run evaluations
    print("\nMemulai evaluasi...")
    
    # 1. Basic metrics
    y_pred = evaluator.evaluate_basic_metrics(X_test, y_test)
    
    # 2. Confusion matrix
    evaluator.plot_confusion_matrix(y_test, y_pred, save_path)
    
    # 3. ROC curves
    evaluator.plot_roc_curves(X_test, y_test, save_path)
    
    # 4. Precision-Recall curves
    evaluator.plot_precision_recall_curves(X_test, y_test, save_path)
    
    # 5. Feature importance
    evaluator.analyze_feature_importance(X_test, y_test, save_path)
    
    # 6. Cross-validation
    evaluator.cross_validation_analysis(X_test, y_test, save_path)
    
    # 7. Prediction confidence
    evaluator.analyze_prediction_confidence(X_test, y_test, save_path)
    
    # 8. Generate report
    evaluator.generate_report(save_path)
    
    print("\n=== EVALUASI SELESAI ===")
    print(f"Semua hasil tersimpan di: {save_path}")

if __name__ == "__main__":
    main()