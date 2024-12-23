# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
#import statsmodels.api as sm
#import altair as alt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from PIL import Image

st.title('Simple Bucket Models')

def load_data():
    #Read hydro data time series
    colp = pd.read_csv('Colpach.txt',index_col='date')
    colp.index = pd.to_datetime(colp.index)
    return colp

def oneevent(colp, event_begin = pd.to_datetime('2012-07-13 14:00:00'), event_end = pd.to_datetime('2012-07-20 14:00:00')):
    #Select one runoff event manulally
    
    event_precip = colp.loc[event_begin:event_end,'precip'].values
    event_runoff = colp.loc[event_begin:event_end,'runoff'].values
    return colp.loc[event_begin:event_end,['precip','runoff']],event_precip, event_runoff

def longevent(colp, event_begin = pd.to_datetime('2012-05-01 14:00:00'), event_end = pd.to_datetime('2012-10-20 14:00:00')):
    #Select one runoff event manulally
    
    event_precip = colp.loc[event_begin:event_end,'precip'].values
    event_runoff = colp.loc[event_begin:event_end,'runoff'].values
    return colp.loc[event_begin:event_end,['precip','runoff']],event_precip, event_runoff


colp = load_data()

eventi = st.checkbox('Lange Zeitreihe')
if eventi:
    colp1, P, Q = longevent(colp)
else:
    colp1, P, Q = oneevent(colp)

#Plot selected event
st.subheader('Niederschlag und Abfluss Daten')
fig = go.Figure()
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Scatter(x=colp1.index,y=colp1.precip, mode='lines', hovertext=colp1.precip, name='Precipitation'),secondary_y=False)
fig.add_trace(go.Scatter(x=colp1.index,y=colp1.runoff, mode='lines', hovertext=colp1.runoff, name='Runoff'),secondary_y=True)
xatitle='time'
yatitle='mm/h'

fig.update_layout(
    xaxis=dict(
        showline=True,
        showgrid=True,
        showticklabels=True,
        title=xatitle,
        linecolor='rgb(204, 204, 204)',
        linewidth=2,
        ticks='outside',
        tickfont=dict(
            family='Arial',
            size=12,
            color='rgb(82, 82, 82)',
        ),
    ),
    yaxis=dict(
        showline=True,
        showgrid=True,
        showticklabels=True,
        title=yatitle,
        linecolor='rgb(204, 204, 204)',
        linewidth=2,
        ticks='outside',
        tickfont=dict(
            family='Arial',
            size=12,
            color='rgb(82, 82, 82)',
        ),
    ),
    autosize=True,
    margin=dict(
        autoexpand=True,
        l=100,
        r=20,
        t=110,
    ),
    showlegend=True,
    template='plotly_white',
    plot_bgcolor='white',
    legend=dict(x=0.02, y=0.98)
)

st.plotly_chart(fig)

st.sidebar.title('Settings')
MO = st.sidebar.selectbox('Model auswählen',['Linear','Beta Store'])
if MO=='Linear':

    image = Image.open('linstore.png')
    st.sidebar.image(image,use_column_width=True)

    st.sidebar.markdown('Mittlere Verweilzeit des Wassers im Speicher:')
    tres = st.sidebar.slider('Residence time (days)', 30, 300, 100)
    st.sidebar.markdown('Verdunstung (als Senkenterm):')
    ETx = st.sidebar.slider('ET (mm/day)', 1., 10., 3.)
    ET = ETx/24.
    st.sidebar.markdown('Initiale Speicherfüllung:')
    sini = st.sidebar.slider('Initial Storage', 1., 10., 2.5)

    #initialisation
    l_event = len(P)
    storage = np.zeros(l_event+1)
    runoff = np.zeros(l_event)
    storage[0] = sini
    
    #Run linear reservoir
    for i in np.arange(l_event): #time step loop (1h)
        runoff[i] = storage[i]/tres
        storage[i+1] = storage[i] - runoff[i] - ET + P[i]
    
    st.subheader(MO+' Model')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=colp1.index,y=runoff, mode='lines', hovertext=colp1.precip, name='Model'))
    fig.add_trace(go.Scatter(x=colp1.index,y=Q, mode='lines', hovertext=colp1.runoff, name='Runoff'))
    
    fig.update_layout(
        xaxis=dict(
            showline=True,
            showgrid=True,
            showticklabels=True,
            title=xatitle,
            linecolor='rgb(204, 204, 204)',
            linewidth=2,
            ticks='outside',
            tickfont=dict(
                family='Arial',
                size=12,
                color='rgb(82, 82, 82)',
            ),
        ),
        yaxis=dict(
            showline=True,
            showgrid=True,
            showticklabels=True,
            title=yatitle,
            linecolor='rgb(204, 204, 204)',
            linewidth=2,
            ticks='outside',
            tickfont=dict(
                family='Arial',
                size=12,
                color='rgb(82, 82, 82)',
            ),
        ),
        autosize=True,
        margin=dict(
            autoexpand=True,
            l=100,
            r=20,
            t=110,
        ),
        showlegend=True,
        template='plotly_white',
        plot_bgcolor='white',
        legend=dict(x=0.02, y=0.98)
    )
    
    st.plotly_chart(fig)

elif MO=='Beta Store':

    image = Image.open('beta.png')
    st.sidebar.image(image,use_column_width=True)

    st.sidebar.markdown('Mittlere Verweilzeit des Wassers im Gewässerspeicher:')
    tres = st.sidebar.slider('Residence time (days)', 30, 300, 100)
    st.sidebar.markdown('beta-Parameter im Bodenspeicher:')
    beta = st.sidebar.slider('beta exponent', 0.3, 3., 1.25)
    st.sidebar.markdown('Größe des Bodenspeichers als Porosität einer 1m Bodensäule:')
    poro = st.sidebar.slider('porosity', 0.2, 0.5, 0.3)
    st.sidebar.markdown('Verdunstung (als Senkenterm):')
    ETx = st.sidebar.slider('ET (mm/day)', 1., 10., 3.)
    ET = ETx/24.
    st.sidebar.markdown('Initiale Speicherfüllung (Gewässer):')
    sini = st.sidebar.slider('Initial Storage', 1., 10., 2.5)
    st.sidebar.markdown('Initiale Speicherfüllung (Boden):')
    tini = st.sidebar.slider('Initial Soil', 100., 300., 150.)

    smax = poro * 1000  #Maximum storage as product of soil porosity and soil depth in mm

    #initialisation
    l_event = len(P)
    storage = np.zeros(l_event+1)
    runoff = np.zeros(l_event)
    storage[0] = sini
    soilmoist = np.zeros(l_event+1)
    soilmoist[0] = tini #Initial soil storage
    subsurf_flow = np.zeros(l_event)
    
    #Run model
    for i in np.arange(l_event): #time step loop (1h)
        #beta store
        subsurf_flow[i] = P[i] * (soilmoist[i]/smax)**beta
        soilmoist[i+1] = soilmoist[i] + P[i] - subsurf_flow[i] - ET
    
        #linear storage
        runoff[i] = storage[i]/tres
        storage[i+1] = storage[i] - runoff[i] + subsurf_flow[i]

    st.subheader(MO+' Model')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=colp1.index,y=runoff, mode='lines', hovertext=colp1.precip, name='Model'))
    fig.add_trace(go.Scatter(x=colp1.index,y=Q, mode='lines', hovertext=colp1.runoff, name='Runoff'))
    
    fig.update_layout(
        xaxis=dict(
            showline=True,
            showgrid=True,
            showticklabels=True,
            title=xatitle,
            linecolor='rgb(204, 204, 204)',
            linewidth=2,
            ticks='outside',
            tickfont=dict(
                family='Arial',
                size=12,
                color='rgb(82, 82, 82)',
            ),
        ),
        yaxis=dict(
            showline=True,
            showgrid=True,
            showticklabels=True,
            title=yatitle,
            linecolor='rgb(204, 204, 204)',
            linewidth=2,
            ticks='outside',
            tickfont=dict(
                family='Arial',
                size=12,
                color='rgb(82, 82, 82)',
            ),
        ),
        autosize=True,
        margin=dict(
            autoexpand=True,
            l=100,
            r=20,
            t=110,
        ),
        showlegend=True,
        template='plotly_white',
        plot_bgcolor='white',
        legend=dict(x=0.02, y=0.98)
    )
    
    st.plotly_chart(fig)

st.sidebar.markdown('Der Code für die Modelle ist im GitHub: https://github.com/cojacoo/simple_bucket')