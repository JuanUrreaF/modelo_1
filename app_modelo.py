import pandas as pd
import streamlit as st
import pickle
import numpy as np

modelo_cargado = pickle.load(open('modelo_rf.pkl', 'rb'))

add_selectbox = st.sidebar.selectbox('',('Pagina principal','Modelo Batch', 'Modelo csv'))

if add_selectbox == 'Pagina principal':
    st.image('cliente.jpg')
    st.title('Modelo prediccion de valor de cliente')
    st.text('Este modelo determina el valor de un cliente a partir de algunas caracteristicas observadas del mismo')

if add_selectbox == 'Modelo Batch':
    st.title('Aqui se puede realizar la prediccion de cada uno de los clientes')
    Age = st.number_input('Ingrese aqui la edad',
        min_value=18,
        max_value=99,
        value=45)
    compras = st.number_input('Ingrese el numero de compras promedio en el mes',
        min_value=0,
        max_value=150,
        value=10)
    gasto = st.number_input('Ingrese el gasto promedio en el mes',
        min_value=0,
        max_value=2000,
        value=20)
    Recency = st.number_input('Numero de dias desde que el cliente no realizo otra transaciion',
        min_value=0,
        max_value=365,
        value=90)
    Income = st.number_input('Salario anual del cliente',
        min_value=15000,
        max_value=200000,
        value=38000)
    input_dict = {'Age':Age, 'compras':compras,'gasto':gasto,'Recency':Recency,'Income':Income}
    input_df = pd.DataFrame([input_dict])

    if st.button('Prediccion'):
        salida = modelo_cargado.predict(input_df)
        output1 = int(salida)
        if output1 == 0:
            st.success('Cliente de valor medio')
        elif output1 == 1:
            st.success('Cliente de alto valor')
        elif output1 == 2:
            st.success('Cliente de bajo valor')
        else:
            st.success('Cliente de muy alto valor')
        st.success(output1)

if add_selectbox == 'Modelo csv':
    archivo_cargado = st.file_uploader('Ingrese aqui el archivo a predecir',type=['csv']) #los corchetes es para especificar que se carga un archivo csv
    if archivo_cargado is not None:
        data_a_predecir = pd.read_csv(archivo_cargado)
        data_a_predecir = data_a_predecir.replace(r'^\s*$', np.nan, regex=True)
        data_a_predecir['Prediccion'] = modelo_cargado.predict(data_a_predecir)
        st.write(data_a_predecir)
