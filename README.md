# 🏨 Hotel Booking Cancellation Prediction

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

**Machine Learning Project | CodeCademy Course**

*Predicting hotel booking cancellations using neural networks*

</div>

---

## 📋 Project Overview

This project focuses on building machine learning models to predict hotel booking cancellations using real-world data from a resort hotel. The dataset contains valuable predictive features including booking dates, length of stay, guest demographics, and pricing information.

### 🎯 **Project Goals**
- Build a **binary classification** model to predict if a customer will cancel their booking
- Build a **multiclass classification** model to predict booking outcomes (Check-out, Canceled, No-Show)
- Provide business value through revenue optimization and resource allocation

### 💼 **Business Impact**
Training well-performing models can help hotel companies:
- 🎯 **Optimize revenue strategies**
- 👥 **Better allocate staff and amenities**
- 📈 **Assist in targeted marketing campaigns**

---

## 👨‍💻 Author

<div align="center">

**Francisco Teixeira Barbosa**

[![GitHub](https://img.shields.io/badge/GitHub-Tuminha-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Tuminha)
[![Email](https://img.shields.io/badge/Email-cisco@periospot.com-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:cisco@periospot.com)
[![Twitter](https://img.shields.io/badge/Twitter-cisco_research-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white)](https://twitter.com/cisco_research)

</div>

---

## 📊 Dataset Information

- **Source**: Kaggle - Hotel Booking Demand Dataset
- **Local Path**: `data/hotel_bookings.csv`
- **Features**: Booking dates, guest demographics, pricing, meal preferences
- **Target Variables**: 
  - `is_canceled` (Binary: 0/1)
  - `reservation_status` (Multiclass: Check-Out/Canceled/No-Show)

---

## 🚀 Project Progress

### Phase 1: Data Exploration & Understanding
<details>
<summary><strong>📈 Import and Inspect</strong></summary>

- [ ] **Task 1**: Import CSV file to pandas DataFrame named `hotels`
- [ ] **Task 2**: Use `.info()` method to inspect data types and missing values
- [ ] **Task 3**: Explore cancellation rates using `.value_counts()` on `is_canceled`
- [ ] **Task 4**: Analyze `reservation_status` column values
- [ ] **Task 5**: Group by `arrival_date_month` and analyze cancellation patterns

</details>

### Phase 2: Data Preprocessing
<details>
<summary><strong>🧹 Data Cleaning and Preparation</strong></summary>

- [ ] **Task 6**: Preview categorical columns with object datatype
- [ ] **Task 7**: Drop irrelevant columns for model training
- [ ] **Task 8**: Label encode `meal` column with meaningful order
- [ ] **Task 9**: Apply one-hot encoding to remaining categorical columns

</details>

### Phase 3: Model Preparation
<details>
<summary><strong>⚙️ Create Training and Testing Sets</strong></summary>

- [ ] **Task 10**: Import PyTorch libraries and modules
- [ ] **Task 11**: Create `train_features` list excluding target variables
- [ ] **Task 12**: Create X and y tensors with proper data types
- [ ] **Task 13**: Split data into 80/20 train/test sets with random_state=42

</details>

### Phase 4: Binary Classification Model
<details>
<summary><strong>🎯 Train Neural Network for Binary Classification</strong></summary>

- [ ] **Task 14**: Build neural network architecture (65→36→18→1 nodes)
- [ ] **Task 15**: Define binary cross-entropy loss and Adam optimizer
- [ ] **Task 16**: Train model for 1000 epochs with performance tracking
- [ ] **Task 17**: Evaluate model on testing set
- [ ] **Task 18**: Calculate accuracy, precision, recall, and F1 scores

</details>

### Phase 5: Multiclass Classification Model
<details>
<summary><strong>🎲 Train Neural Network for Multiclass Classification</strong></summary>

- [ ] **Task 19**: Label encode `reservation_status` categories
- [ ] **Task 20**: Create X and y tensors for multiclass problem
- [ ] **Task 21**: Split data into train/test sets
- [ ] **Task 22**: Build multiclass neural network (65→65→36→3 nodes)
- [ ] **Task 23**: Define cross-entropy loss and Adam optimizer
- [ ] **Task 24**: Train model for 500 epochs
- [ ] **Task 25**: Evaluate multiclass model on testing set
- [ ] **Task 26**: Calculate comprehensive classification metrics

</details>

---

## 🛠️ Technical Stack

<div align="center">

| Technology | Purpose | Version |
|------------|---------|---------|
| **Python** | Programming Language | 3.11+ |
| **Pandas** | Data Manipulation | Latest |
| **PyTorch** | Neural Networks | Latest |
| **Scikit-learn** | Model Evaluation | Latest |
| **NumPy** | Numerical Computing | Latest |

</div>

---

## 📁 Project Structure

```
predict_hotel_cancellations/
├── 📁 data/
│   └── hotel_bookings.csv
├── 📓 hotel_cancellations.ipynb
└── 📄 README.md
```

---

## 🎓 Learning Objectives

By completing this project, you will:

- ✅ **Master PyTorch fundamentals** for neural network development
- ✅ **Understand binary vs multiclass classification** approaches
- ✅ **Learn proper data preprocessing** techniques for ML
- ✅ **Implement model evaluation** with comprehensive metrics
- ✅ **Gain experience with real-world datasets** and business applications

---

## 🚦 Getting Started

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd predict_hotel_cancellations
   ```

2. **Install dependencies**
   ```bash
   pip install pandas torch scikit-learn numpy
   ```

3. **Download dataset** (if needed)
   ```python
   from hotel_cancellations import download_kaggle_dataset
   data_path = download_kaggle_dataset("jessemostipak/hotel-booking-demand")
   ```

4. **Start learning!** 🎯
   - Open `hotel_cancellations.ipynb`
   - Follow the tasks in order
   - Check off completed tasks in this README

---

## 📈 Expected Outcomes

### Binary Classification Model
- **Target**: Predict cancellation (Yes/No)
- **Architecture**: 65 → 36 → 18 → 1 nodes
- **Training**: 1000 epochs
- **Evaluation**: Accuracy, Precision, Recall, F1

### Multiclass Classification Model
- **Target**: Predict booking outcome (Check-out/Canceled/No-show)
- **Architecture**: 65 → 65 → 36 → 3 nodes
- **Training**: 500 epochs
- **Evaluation**: Comprehensive classification report

---

## 🎯 Tips for Success

<div align="center">

### 💡 **Learning Approach**
- **Take your time** - Don't rush through the tasks
- **Experiment** - Try different hyperparameters
- **Document** - Keep notes on what works
- **Ask questions** - Use hints when needed

</div>

---

## 📚 Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [CodeCademy Machine Learning Course](https://www.codecademy.com/learn/machine-learning)

---

<div align="center">

**Happy Learning! 🚀**

*Remember: The journey of a thousand models begins with a single dataset*

</div>
