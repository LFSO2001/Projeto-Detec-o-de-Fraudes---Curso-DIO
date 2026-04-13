#Projeto - Detecção de Fraudes em Transações Bancárias
#importação da biblioteca Pandas
import pandas as pd

#URL do dataset das transações para treinamento
url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"

df = pd.read_csv(url)

print(df.head()) #para ver se importou o dataset e mostra as primeiras linhas

#Cada limha é uma transação 
#Coluna time tempo da transação
#Coluna v1 a v28 -> Variaveis transformadas (presrva a privacidade e carregam padrões das transações)
#Coluna Amount -> Valor das Transações
#Coluna Class -> 0: não fraudulenta e 1: fraudulenta

#Classificação desbalanceada, pois uma classe aparesse em uma proporção muito maior que a outra
print(df["Class"].value_counts(normalize=True)) #Calcula a proporção de cada tipo de transação

#Feature Engeneering
#Criar novas ou transformar variáveis que ajudam no modelo
import numpy as np

df["Amount_log"] = np.log1p(df["Amount"]) #loglp -> reduz a diferença entre valores e deixa a distribuição dos dados melhor

#Padronização da escala
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

df["Amount_scaled"] = scaler.fit_transform(df[["Amount"]])

from sklearn.model_selection import train_test_split

x = df.drop("Class",axis=1) # -> remove a coluna Class
y = df["Class"]

x_train,x_test,y_train,y_test = train_test_split(x,y,stratify=y,test_size=0.3,random_state=42) #Separação do modelo, divide os dados em conjunto de treino e de teste.
#test_size -> 30% dos dados vão para teste
#stratify -> Mantem a classe fraudulenta e não fraudulenta
 
#Modelo de Classificação - Regressão Logística - prever características
from sklearn.linear_model import LogisticRegression 

model = LogisticRegression(max_iter=1000) #1000 iterações

model.fit(x_train,y_train) # Modelo aprende padrões pelos dados de treino 

y_pred = model.predict(x_test) # Modelo faz previsão nos dados de teste (dados que o modelo ainda não viu)

from sklearn.metrics import classification_report

print(classification_report(y_test,y_pred)) # Ver as métricas mais importantes para a avaliação


from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

y_probs = model.predict_proba(x_test)[:,1]

fpr, tpr, _ = roc_curve(y_test,y_probs)

plt.plot(fpr,tpr)
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()

print(f"AUC: {roc_auc_score(y_test,y_probs)}")

from sklearn.metrics import precision_recall_curve

precision, recall, _ = precision_recall_curve(y_test,y_probs)

plt.plot(recall,precision)
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.show()

#Balanceamento de dados
#Undersampling -> reduz a classe majoritaria para o tamanho da minoritaria. Reduz o numero de transações normais para o mesmo das fraudulentas
#Cria um dataset equilibrado, modelo aprende melhor a identificar as fraudes
#O problema é que "perde" muitos dados de transações normais, podendo reduzir a qualidade do modelo
fraudes = df[df["Class"] == 1]
normais = df[df["Class"] == 0].sample(len(fraudes),random_state=42)

df_under = pd.concat([fraudes,normais])

#Oversampling -> cria novos exemplos sinteticos da classe minoritária
#ponto positivo -> não perde dados
#problema -> inclui dados artificiais
from imblearn.over_sampling import SMOTE

smote = SMOTE()

x_res,y_res = smote.fit_resample(x,y)

#para saber qual modelo desses utilizar, precisamos testar e ver qual apresenta o melhor resultado


from sklearn.ensemble import RandomForestClassifier #Modelo baseado em arvores de decisão, conjuntos de arvores de decisão combinados

rf = RandomForestClassifier(
    n_estimators=50,
    max_depth=10,
    class_weight="balanced",
     n_jobs=-1, #ajustando automatimente o peso das classes
    random_state=42
    )

rf.fit(x_train,y_train)

y_pred_rf = rf.predict(x_test)

print(classification_report(y_test,y_pred_rf))

#Pipeline -> Organizar fluxo de processamento
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model",LogisticRegression(max_iter=1000))
])

pipeline.fit(x_train, y_train)
y_pred = pipeline.predict(x_test)

threshold = 0.3 #Probabilidade de fraude maior que 0.3 ele classifica como fraude -> Queremos aumentar o Recall

y_pred_custom = (y_probs > threshold).astype(int)
print(classification_report(y_test,y_pred_custom))

#Modelo Avançado XGBoost -> Mais poderoso que o random forest
#Varios modelos simples treinados em sequencia, o novo corrige o anterior até o final (aprendizado sequencial)
from xgboost import XGBClassifier

xgb = XGBClassifier(
    scale_pos_weight = 10,
    use_label_encoder=False,
    eval_metric = "logloss"
)

xgb.fit(x_train,y_train)
y_pred_xgb = xgb.predict(x_test)
print(classification_report(y_test,y_pred_xgb))

#Importancia das variaveis 
import matplotlib.pyplot as plt
importancias = xgb.feature_importances_

plt.bar(range(len(importancias)),importancias)
plt.title("importância das variáveis")
plt.show()

#Ajuste de HiperParametros
#Testar varias combinações de parâmetros pra melhorar o modelo
from sklearn.model_selection import GridSearchCV

param_grid = {   #conjunto de combinações para testar
    "max_depth": [3,5],
    "n_estimators": [50,100]
}

grid = GridSearchCV(  #testa todas as combinações entre os valores
    XGBClassifier(eval_metric="logloss"),
    param_grid,
    scoring = "recall",
    cv=3
)

grid.fit(x_train,y_train)

print("Melhor modelo:",grid.best_params_)

#Explicabilidade
#Como cada variavel contribui para o modelo
#permite entender o modelo
import shap
explainer = shap.Explainer (xgb)
shap_values = explainer(x_test[:100])
shap.plots.bar(shap_values)