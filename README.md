# Indian Used Bike Price Prediction – Machine Learning Project

> **A data-driven ML system that predicts the resale price of used bikes based on key features like brand, city, kilometers driven, engine power, and ownership history.**

---
<img width="870" height="480" alt="image" src="https://github.com/user-attachments/assets/958d3a40-516b-471c-bc83-9a107dd51a49" />

---

## Problem Statement
- The pricing of used bikes in India is inconsistent and depends on numerous factors.
- Buyers often get confused, and sellers struggle to estimate a fair value.  

## Solution: 
- A complete Machine Learning pipeline that analyzes past data of 32,000+ used bikes and predicts an accurate resale price using advanced regression models.
- The system cleans data, handles outliers, engineers new features, and evaluates multiple models to deliver the best prediction.

---

## 🛠️ Technologies Used
- **Python**
- **Pandas & NumPy**
- **Matplotlib & Seaborn**
- **Scikit-learn**
- **XGBoost regressor**
- **Random Forest Regressor**
- **Linear Regressor**
- **Decision tree Regressor**
---

## Dataset Information: 
- **Collected from Kaggle and scraped from droom.in** 
- **Contains bike name, brand, city, km driven, number of owners, power, age, and price** 
- [Dataset Link](https://www.kaggle.com/datasets/saisaathvik/used-bikes-prices-in-india)
---

## Key Features
- bike_name
- brand
- city
- kms_driven
- owner
- age
- power
- **price (Target)**
---

## Wow Factors
- End-to-end automated data preprocessing
- Feature engineering: bike age, km per year
- Outlier handling using IQR
- Machine learning with 4 regression models
- Detailed performance comparison
- Prediction-ready trained models saved as .joblib
- Visual output graphs for clarity
---

## End Users
- Bike sellers & dealers
- Customers / buyers
- ML enthusiasts
- Data science learners
- Automobile market analysts
---

## Results

### 🔹 Result 1 — Data Preprocessing Output

<img width="1150" height="699" alt="image" src="https://github.com/user-attachments/assets/6c60af8f-969d-48fe-9f35-6e05873d266d" />
_______________________________________________________________________________________________________________________________________________________
<img width="1586" height="853" alt="image" src="https://github.com/user-attachments/assets/c452754a-b319-4d8a-a610-fd0d0b99e1d5" />
_______________________________________________________________________________________________________________________________________________________
<img width="1259" height="745" alt="image" src="https://github.com/user-attachments/assets/14eda871-8c26-4570-8a68-3446d68b5a5b" />
_______________________________________________________________________________________________________________________________________________________

###  🔹 Result 2 — Feature Engineering / Normalization

<img width="1279" height="286" alt="image" src="https://github.com/user-attachments/assets/7172ba8e-fccb-4ff6-8d40-4fd51388d31f" />
_____________________________________________________________________________________________________________________________________________________

### 🔹 Result 3 — Remove Outliers
<img width="1395" height="825" alt="image" src="https://github.com/user-attachments/assets/364d74e5-2ef5-4865-b252-8932e0ee3592" />
_____________________________________________________________________________________________________________________________________________________
<img width="1510" height="467" alt="image" src="https://github.com/user-attachments/assets/857f62f8-1f77-4252-a728-646826ebca62" />
_____________________________________________________________________________________________________________________________________________________

### 🔹 Result 4 — Model Performance Comparison

<img width="929" height="270" alt="image" src="https://github.com/user-attachments/assets/ebc97027-474b-425d-b63d-1c1047ce3fa4" />
_______________________________________________________________________________________________________________________________________________________
### 🔹 Result 5 — Actual vs Predicted Graph
<img width="1463" height="689" alt="image" src="https://github.com/user-attachments/assets/fb55c27d-d139-431a-bbd2-963bc5c9f8be" />
_______________________________________________________________________________________________________________________________________________________
<img width="1450" height="688" alt="image" src="https://github.com/user-attachments/assets/ebe77e9b-bf54-4c12-9268-f566bf33474d" />
_______________________________________________________________________________________________________________________________________________________
<img width="1448" height="682" alt="image" src="https://github.com/user-attachments/assets/07f09e9e-fd9e-442e-8382-07235fdab11b" />
_______________________________________________________________________________________________________________________________________________________
---

## Conclusion
In this used bike price dataset, all four models were tested, but **Random Forest** gave the highest accuracy. This is because:

- **The dataset is non-linear, and Random Forest handles non-linear patterns better than Linear Regression.**
- **The data contains outliers and noisy values (price, kms, power), which Random Forest manages more effectively.**
- **There are complex feature interactions (city × brand, age × kms_driven), and Random Forest captures them automatically.**
- **Decision Tree alone overfit the data, while Random Forest reduces overfitting by combining many trees.**
- **XGBoost performed well but needs more hyperparameter tuning; with default settings, it was slightly weaker than Random Forest.**
---

## Future Scope
- Add more bike details (service history, insurance, condition)
- Include hyperparameter tuning for even better accuracy
- Visual dashboard for bike dealers
---

## 🔗 Useful Links

- [Kaggle Dataset](https://www.kaggle.com/datasets/saisaathvik/used-bikes-prices-in-india)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/en/stable/)
- [Random Forest Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
- [Python Documentation](https://docs.python.org/3.15/)

---
🔗 [Connect with me on LinkedIn](https://www.linkedin.com/in/vaishnavi-shinde-40190b2b1/)

---

## Author
**Shinde Vaishnavi**  
**202402060016**

MIT Academy of Engineering, Electronics Engineering  
> It is a group project Created during Computational intelligence project examination
---

