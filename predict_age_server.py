from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager
import uvicorn
import io
import base64
import cv2
from mtcnn import MTCNN
import numpy as np

from pydantic import BaseModel
import csv
import os

import os, sys
# Force the C++ backend to shut up BEFORE it boots
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import warnings
import time
import math

import pickle

from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import seaborn as sns
import plotly.graph_objects as go
from IPython.display import Image, display

import lifelines
from lifelines.utils import concordance_index
from lifelines.statistics import logrank_test
from sksurv.util import Surv
from sksurv.metrics import concordance_index_ipcw

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import tensorflow_probability as tfp

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config = config)

from tensorflow import keras
from tensorflow.keras import optimizers, initializers, regularizers, layers

import scipy.stats as stats
from scipy.stats import norm, t, probplot, pearsonr, spearmanr, rankdata
from scipy.stats import truncnorm as truncnorm_scipy
from scipy.stats import gamma as gamma_dist
from scipy.special import gamma

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold

import thetaflow as thf
import truncnorm

# Add utils file to path so we can import it
import gndr_utils as utils

app = FastAPI()

# Detector de rostos MTCNN
detector = MTCNN()

# Dicionário global para os modelos carregados
ml_models = {}

# Comprimento dos intervalos - Considerando o intervalo 1-sigma da normal
# Para aplicações mais rigorosas, o ideal é considerar 2-sigma (em torno de 95% de confiança para a idade)
# q1 = 0.025
q1 = norm.cdf(-1)
# q2 = 0.975
q2 = norm.cdf(1)

def build_utkface_model( dropout_rate = 0.3 ):
    tn_parameters = {
        "mu": {"link": tf.math.exp, "link_inv": tf.math.log, "par_type": "nn",
               "shape": 1, "init": 1.0, "warmup_time": 0},
        "sigma": {"link": tf.math.exp, "link_inv": tf.math.log, "par_type": "nn", "shape": 1,
                  "init": 1.0, "warmup_time": 0}
    }

    def tn_loglikelihood_loss(model, nn_output, data):
        X, y = data
        
        # Extract variables from the neural network output
        mu = model.get_variable("mu", nn_output)
        sigma = model.get_variable("sigma", nn_output)

        # Flatten to ensure perfect dimension alignment
        y = tf.reshape(y, [-1])
        mu = tf.reshape(mu, [-1])
        sigma = tf.reshape(sigma, [-1])
        
        # Route directly into the custom analytical gradient function.
        # TensorFlow will use @tf.custom_gradient definitions under the hood
        # rather than trying to AutoDiff through the truncation boundaries.
        loglik_tensor = truncnorm.log_pdf(y, mu, sigma)
        
        # Return the Negative Log-Likelihood (NLL)
        return -tf.reduce_sum( loglik_tensor )

    def tn_neural_network(model, seed = None):
        initializer = initializers.GlorotNormal(seed = seed)
        
        model.feature_extractor = tf.keras.applications.MobileNetV2(
            input_shape = (200, 200, 3),
            include_top = False,
            weights = 'imagenet',
            pooling = 'avg'
        )

        # We train the MobileNetV2 weights only from the 100th layer forward.
        # By fine-tuning that model, we are able to obtain more granular and specific traits
        # particularly related to our problem and corresponding loss function
        model.feature_extractor.trainable = True
        for layer in model.feature_extractor.layers[:100]:
            layer.trainable = False

        model.dropout = keras.layers.Dropout(dropout_rate)

        model.dense1 = keras.layers.Dense(
            units = 128, 
            activation = tf.nn.gelu, 
            kernel_initializer = initializer,
            dtype = tf.float32,
            name = "flattened_layer"
        )
        
        model.dense2 = keras.layers.Dense(
            units = 64, 
            activation = tf.nn.gelu, 
            kernel_initializer = initializer,
            dtype = tf.float32,
            name = "LLLA_layer"
        )

        model.dense3 = keras.layers.Dense(
            units = 2, 
            activation = None, 
            use_bias = True,
            bias_initializer = initializer,
            dtype = tf.float32,
            name = "normal_output"
        )
    
    def tn_neural_network_call(model, x_input, training = False):
        features = model.feature_extractor(x_input, training = training)
        x = model.dropout(features, training = training)
        x = model.dense1(x)
        x = model.dense2(x)
        output = model.dense3(x)
        return output
    
    def tn_neural_network_call_nolast(model, x_input):
        features = model.feature_extractor(x_input, training = False) 
        x = model.dropout(features, training = False)
        x = model.dense1(x)
        x = model.dense2(x)
        return x

    return tn_parameters, tn_loglikelihood_loss, tn_neural_network, tn_neural_network_call, tn_neural_network_call_nolast

def log_tn_lower_quantile(model, nn_output, data):
    mu = model.get_variable("mu", nn_output)
    sigma = model.get_variable("sigma", nn_output)
    
    # Standard Normal baseline
    norm_dist = tfp.distributions.Normal(loc = 0.0, scale = 1.0)
    
    # Compute alpha = Phi(-mu/sigma)
    alpha = norm_dist.cdf(-mu / sigma)
    
    # Adjust the 2.5% target for the [0, inf] truncation
    p_target = q1
    adjusted_p = alpha + p_target * (1.0 - alpha)
    
    # Clip the probability slightly inside [0, 1] 
    # Prevents GradientTape from hitting the ndtri singularities
    adjusted_p_safe = tf.clip_by_value(adjusted_p, 1e-7, 1.0 - 1e-7)
    
    # Map back through the inverse CDF and scale
    q_025 = mu + sigma * norm_dist.quantile(adjusted_p_safe)
    
    # Return the log to match theoretical framework
    return tf.math.log(q_025)

def log_tn_upper_quantile(model, nn_output, data):
    mu = model.get_variable("mu", nn_output)
    sigma = model.get_variable("sigma", nn_output)
    
    norm_dist = tfp.distributions.Normal(loc = 0.0, scale = 1.0)
    
    alpha = norm_dist.cdf(-mu / sigma)
    
    p_target = q2
    adjusted_p = alpha + p_target * (1.0 - alpha)
    
    adjusted_p_safe = tf.clip_by_value(adjusted_p, 1e-7, 1.0 - 1e-7)
    q_975 = mu + sigma * norm_dist.quantile(adjusted_p_safe)
    
    return tf.math.log(q_975)

def process_and_crop_face(image_bytes):
    img_tf = tf.image.decode_jpeg(image_bytes, channels=3)
    img_np = img_tf.numpy()
    
    resultados = detector.detect_faces(img_np)
    
    if len(resultados) == 0:
        return None

    rosto_principal = max(resultados, key=lambda b: b['confidence'])
    x, y, width, height = rosto_principal['box']

    # Previne valores negativos do MTCNN
    x, y = max(0, x), max(0, y)

    # Encontra o centro exato do rosto
    centro_x = x + width // 2
    centro_y = y + height // 2
    
    # Transforma em quadrado pegando a maior dimensão
    lado_maximo = max(width, height)
    
    # Multiplica por 1.6 para dar margem (afastar o zoom)
    fator_margem = 1.6
    tamanho_quadrado = int(lado_maximo * fator_margem)
    
    # 4. Calcula as novas coordenadas X, Y do topo superior esquerdo
    novo_x = centro_x - (tamanho_quadrado // 2)
    novo_y = centro_y - (tamanho_quadrado // 2)
    
    # 5. Garante que o recorte não vai pedir pixels fora da foto (padding natural)
    img_height, img_width, _ = img_np.shape
    x1 = max(0, novo_x)
    y1 = max(0, novo_y)
    x2 = min(img_width, novo_x + tamanho_quadrado)
    y2 = min(img_height, novo_y + tamanho_quadrado)
    
    largura_corte = x2 - x1
    altura_corte = y2 - y1

    # Recorta usando TensorFlow
    pic_crop = tf.image.crop_to_bounding_box(
        img_tf, 
        offset_height=y1, 
        offset_width=x1, 
        target_height=altura_corte, 
        target_width=largura_corte
    )

    pic_resize = tf.image.resize(pic_crop, [200, 200])

    pic_normalized = tf.cast(pic_resize, tf.float32)
    pic_normalized = (pic_normalized / 127.5) - 1.0
    
    return pic_normalized
    
def predict_age(img):
    alpha = 0.05
    z_norm = norm.ppf(1-alpha/2)
    
    utkfaces_model = ml_models["utkfaces_model"]
    
    exp_img = tf.expand_dims(img, axis = 0)
    utkfaces_model.predict(exp_img)

    img_lower_quantile = utkfaces_model.variable_function_covariance(
        fun = log_tn_lower_quantile,
        data = [],
        x = exp_img
    ).numpy().flatten()
    
    img_upper_quantile = utkfaces_model.variable_function_covariance(
        fun = log_tn_upper_quantile,
        data = [], 
        x = exp_img
    ).numpy().flatten()
    
    img_pred = utkfaces_model.predict(exp_img)
    
    mu_pred = img_pred["mu"]
    sigma_pred = img_pred["sigma"]

    tnorm_dist_test = tfp.distributions.TruncatedNormal(loc = mu_pred, scale = sigma_pred, low = 0.0, high = np.inf)
    img_lower_quantile_hat = np.log( tnorm_dist_test.quantile( q1 ) )
    img_upper_quantile_hat = np.log( tnorm_dist_test.quantile( q2 ) )
    
    img_lower_log_tn_lower_quantile = img_lower_quantile_hat - np.sqrt(img_lower_quantile) * z_norm
    img_upper_log_tn_upper_quantile = img_upper_quantile_hat + np.sqrt(img_upper_quantile) * z_norm
    img_lower_tn_lower_quantile = np.exp( img_lower_log_tn_lower_quantile )
    img_upper_tn_upper_quantile = np.exp( img_upper_log_tn_upper_quantile )
    
    return mu_pred, sigma_pred, img_lower_tn_lower_quantile, img_upper_tn_upper_quantile
    

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Este código irá ser executado assim que o servidor for inicializado
    
    # Carrega o modelo e os seus pesos treinados
    with tf.device("/GPU:0"):
        tn_parameters, tn_loglikelihood_loss, tn_neural_network, tn_neural_network_call, tn_neural_network_call_nolast = \
        build_utkface_model( dropout_rate = 0.3 )
        seed = 10
        utkfaces_model = thf.ModelNN(tn_parameters, tn_loglikelihood_loss,
                                     tn_neural_network, tn_neural_network_call,
                                     tn_neural_network_call_nolast, input_dim = (200,200,3), seed = seed)
        utkfaces_model.load_model("whole_sample_training_mobilenet_layers")
    
    # Salve o modelo instanciado no dicionário global
    ml_models["utkfaces_model"] = utkfaces_model
    
    print("Modelo carregado na memória e pronto para predições.")

    # Congela o servidor e fica aguardando pelas submissões
    yield 

    print("Limpando memória e desligando servidor...")
    ml_models.clear()


app = FastAPI(lifespan=lifespan)

@app.get("/")
async def root():
    # Lê o arquivo HTML e o envia como resposta da página principal
    with open("index.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    img = process_and_crop_face(contents)

    if(img is None):
        return {"erro": "Nenhum rosto detectado na foto. Tente novamente."}
    
    mu_pred, sigma_pred, lower_quantile_pred, upper_quantile_pred = predict_age(img)
    mensagem = "SUCESSO!"

    mu_pred = mu_pred.numpy()[0,0]
    sigma_pred = sigma_pred.numpy()[0,0]
    lower_quantile_pred = lower_quantile_pred[0,0]
    upper_quantile_pred = upper_quantile_pred[0,0]

    ts = np.linspace(0.001, 100, 500)
    
    a_std = (0.0 - mu_pred) / sigma_pred
    norm_ts = truncnorm_scipy.pdf(ts, loc = mu_pred, scale = sigma_pred, a = a_std, b = np.inf)

    fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (12,6))

    print("img shape", img.shape)

    img_plot = (img - tf.math.reduce_min(img)) / (tf.math.reduce_max(img) - tf.math.reduce_min(img))
    
    ax[0].imshow( img_plot )
    ax[0].axis('off')
    ax[1].plot(ts, norm_ts, label = "Age distribution", color = "blue")
    ax[1].axvline(lower_quantile_pred, color = "blue", linestyle = "dashed", label = "Predictive interval")
    ax[1].axvline(upper_quantile_pred, color = "blue", linestyle = "dashed")
    fig.tight_layout()
    
    # Cria o buffer de memória
    buffer = io.BytesIO()
    # Salva a figura no buffer (bbox_inches='tight' remove bordas brancas desnecessárias)
    fig.savefig(buffer, format = "png", bbox_inches = "tight")
    # Codifica os bytes para base64 e converte para string
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    # Fecha a figura para não estourar a memória do servidor a cada requisição
    plt.close(fig)
    
    return {
        "mensagem": mensagem,
        "media": str(mu_pred),
        "desvio_padrao": str(sigma_pred),
        "intervalo_inferior": str(lower_quantile_pred),
        "intervalo_superior": str(upper_quantile_pred),
        "imagem_base64": img_base64
    }

class FeedbackIdade(BaseModel):
    idade_predita: float
    idade_real: float
    idade_predita_lower: float
    idade_predita_upper: float

@app.post("/feedback")
async def receber_feedback(dados: FeedbackIdade):
    arquivo_csv = "feedbacks/resultados_congresso.csv"
    
    arquivo_existe = os.path.exists(arquivo_csv)
    
    with open(arquivo_csv, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not arquivo_existe:
            writer.writerow([
                "idade_predita", 
                "idade_real", 
                "erro_absoluto", 
                "limite_inferior", 
                "limite_superior", 
                "dentro_do_intervalo"
            ])
            
        erro = abs(dados.idade_predita - dados.idade_real)
        
        # Verifica se a idade real caiu dentro do intervalo preditivo (1 = Sim, 0 = Não)
        cobertura = 1 if (dados.idade_predita_lower <= dados.idade_real <= dados.idade_predita_upper) else 0
        
        writer.writerow([
            round(dados.idade_predita, 2), 
            dados.idade_real, 
            round(erro, 2),
            round(dados.idade_predita_lower, 2),
            round(dados.idade_predita_upper, 2),
            cobertura
        ])
        
    return {"mensagem": "Salvo com sucesso"}


# Roda o servidor se o script for executado diretamente
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port = 8000)