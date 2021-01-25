# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 10:10:30 2020

@author: William.Benn
"""

import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
from dash.dependencies import Input, Output
import plotly.express as px
import pickle
from sklearn.calibration import CalibratedClassifierCV
from catboost import CatBoostClassifier
import gunicorn

## Import scaling

scaling = pd.read_csv("Data/xscaling.csv")
medians = pd.read_csv("Data/medians.csv")
scaling.index = ["Mean", "Standard Deviation"]

## Import model

with open(f'Model/model2.pkl', 'rb') as f:
    calibrated = pickle.load(f)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

question_width = "420px"
question_color = "#355C7D"
data_entry_color = "white"
slider_font_size = "13px"
slider_width = "350px"
slider_padding = "35px"
text_left_margin = "30px"
text_size = "13px"

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

app.layout = html.Div([
    
        html.Div([
            
            html.Div([
                
                html.H1("Predicting affinity with UK political parties",
                style = {'textAlign' : 'center', 
                         "font-weight" : "bold", 
                         "padding-top" : "40px", 
                         "padding-bottom" : "35px",
                         "margin-bottom" : "0px"})
                
                ],
                style = {#"background-image" : "linear-gradient(to right, #94b5d1, #355C7D, #355C7D, #355C7D, #355C7D, #355C7D, #94b5d1)",
                         "color" : "white",
                         "background-color" : "#355C7D"}),
            
            html.Div([
                
               html.Div([
                   
                    html.H2("This study uses UK election survey data to predict which political party people feel closest to",
                    style = {'textAlign' : 'center', 
                             'width': "600px", 
                             "font-weight" : "bold", 
                             "font-size" :"22px"})
                    
                    ], 
                   style = {"display" : "flex", 
                            "align-items": "center", 
                            "justify-content": "center",
                            "padding-bottom" : "10px"}), 
               
                html.Div([
                    
                    html.P("The ten questions below have been selected from the initial survey question set for their predictive power. Modify the answers and check the graph to see how various responses affect the probability that someone would feel closest to each political party.",
                    style = {'textAlign' : 'center', 
                             'width': "750px", 
                             "font-size" :"16px"})
                    
                    ], style = {"display" : "flex", 
                            "align-items": "center", 
                            "justify-content": "center"}),
                
                ], 
                style = {#"background-image" : "linear-gradient(to right,  #fccfd4, #F67280, #F67280, #F67280, #F67280, #F67280, #fccfd4)", 
                         "color" : "white",
                         "background-color" : "#F67280",
                         "padding-top" : "25px",
                         "padding-bottom" : "25px"}),
                            
            ], 
            style = {"width" : "100%"}),
 
    html.Div([
       
        html.Div([
            
            html.Div([
                
                    html.P("Change the answers below to see how different responses affect predictions",
                    style = {"font-size" : 18,
                             "font-weight" : "bold",
                             "font-color" : "#333333",
                             "margin-left" : text_left_margin,
                             "margin-right" : text_left_margin}),
                
                ],
                style = {"width" : question_width,
                         "background-color" : "white",
                         "padding-top" : "5px",
                         "padding-bottom" : "15px"}),
                            
            html.Div([
                
                html.Div([
                    
                    html.P("1. Age",
                           style = {"margin-right" : "52px",
                                    "margin-left" : text_left_margin,
                                    "font-size" : text_size,
                                    "padding-top" : "7px"}),
                    
                    dcc.Input(
                        id = "age",
                        value = 30,
                        min=18, 
                        max=99,
                        type='number',
                        style = {'width': "70px",
                                 "font-size" : text_size}),
                    
                    html.P("(between 18 and 99)",
                           style = {"margin-right" : "52px",
                                    "margin-left" : "15px",
                                    "font-size" : text_size,
                                    "font-style" : "italic",
                                    "padding-top" : "7px"}),
                    
                
                
                    ],
                    style = {"width" : question_width,
                             "background-color" : "white",
                             "display" : "flex",
                             "align-items": "right",
                             "justify-content": "right",
                             "padding-top" : "0px"}),
                
                ], 
                style = {"display" : "flex", 
                         "align-items": "center", 
                         "justify-content": "center",
                         "padding-bottom" : "2px"}),
            
            
            html.Div([
                
                html.Div([
                
                    html.P("2. Region",
                           style = {"margin-right" : "30px",
                                    "margin-left" : text_left_margin,
                                    "font-size" : text_size,
                                    "padding-top" : "7px"}),    
                
                    dcc.Dropdown(
                        id='region',
                        options=[
                            {'label': "North East", 'value': "north east"},
                            {'label': "North West", 'value': "north west"},
                            {'label': "Yorkshire and the Humber", 'value': "yorkshire and the humber"},
                            {'label': "East Midlands", 'value': "east midlands"},
                            {'label': "West Midlands", 'value': "west midlands"},
                            {'label': "East of England", 'value': "east of england"},
                            {'label': "London", 'value': "london"},
                            {'label': "South East", 'value': "south east"},
                            {'label': "South West", 'value': "south west"},
                            {'label': "Wales", 'value': "wales"},
                            {'label': "Scotland", 'value': "scotland"}],
                        value ='london',
                        clearable = False,
                        style = {'width': "260px",
                                 "font-size" : text_size,
                                 "padding-left" : "3px"})
                
                    ],
                    style = {"width" : question_width,
                             "background-color" : "white",
                             "padding-top" : "13px",
                             "padding-bottom" : "5px",
                             "display" : "flex",
                             "align-items": "right",
                             "justify-content": "right"}),
                
                
                ], 
                style = {"display" : "flex", 
                         "align-items": "center", 
                         "justify-content": "center",
                         "padding-bottom" : "8px"}),
            
         
            html.Div([
                
                    html.P("3. Trust in British politicians (0 = no trust)",
                    style = {"font-size" : text_size,
                             "margin-left" : text_left_margin}),
                
                ],
                style = {"width" : question_width,
                         "background-color" : "white",
                         "padding-top" : "5px",
                         "padding-bottom" : "0px"}),
                
             
         
            html.Div([
                
                html.Div([
                
                    html.Div([
                
                            dcc.Slider(
                                id = "trust",
                                value = 5,
                                min = 0,
                                max = 10,
                                marks={0: {'label': '0', 'style': {'font-size': slider_font_size}},
                                       1: {'label': '1', 'style': {'font-size': slider_font_size}},
                                       2: {'label': '2', 'style': {'font-size': slider_font_size}},
                                       3: {'label': '3', 'style': {'font-size': slider_font_size}},
                                       4: {'label': '4', 'style': {'font-size': slider_font_size}},
                                       5: {'label': '5', 'style': {'font-size': slider_font_size}},
                                       6: {'label': '6', 'style': {'font-size': slider_font_size}},
                                       7: {'label': '7', 'style': {'font-size': slider_font_size}},
                                       8: {'label': '8', 'style': {'font-size': slider_font_size}},
                                       9: {'label': '9', 'style': {'font-size': slider_font_size}},
                                       10: {'label': '10', 'style': {'font-size': slider_font_size}}})
                         
                            ],
                            style = {"width" : slider_width,
                                     "padding-top" : "5px",
                                     "padding-bottom" : "10px",
                                     "padding-left" : slider_padding,
                                     "padding-right" : slider_padding,
                                     "background-color" : "white"})
                                     #"border-style" : "solid",
                                     #"border-color" : "#CCCCCC",
                                     #"border-width" : "1px",
                                     #"border-radius" : "5px"
               
                    ],
                    style = {"width" : question_width,
                             "background-color" : "white",
                             "padding-bottom" : "20px",
                             "padding-top" : "0px"}),
                
                ], 
                style = {"display" : "flex", 
                         "align-items": "center", 
                         "justify-content": "center",
                         "padding-bottom" : "8px"}),
            
            html.Div([
                
                    html.P("4. Consider yourself left or right wing (0 = left wing)",
                    style = {"font-size" : text_size,
                             "margin-left" : text_left_margin}),
                
                ],
                style = {"width" : question_width,
                         "background-color" : "white",
                         "padding-top" : "5px",
                         "padding-bottom" : "0px"}),
                
             
         
            html.Div([
                
                html.Div([
                
                    html.Div([
                
                            dcc.Slider(
                                id = "left-right",
                                value = 5,
                                min = 0,
                                max = 10,
                                marks={0: {'label': '0', 'style': {'font-size': slider_font_size}},
                                       1: {'label': '1', 'style': {'font-size': slider_font_size}},
                                       2: {'label': '2', 'style': {'font-size': slider_font_size}},
                                       3: {'label': '3', 'style': {'font-size': slider_font_size}},
                                       4: {'label': '4', 'style': {'font-size': slider_font_size}},
                                       5: {'label': '5', 'style': {'font-size': slider_font_size}},
                                       6: {'label': '6', 'style': {'font-size': slider_font_size}},
                                       7: {'label': '7', 'style': {'font-size': slider_font_size}},
                                       8: {'label': '8', 'style': {'font-size': slider_font_size}},
                                       9: {'label': '9', 'style': {'font-size': slider_font_size}},
                                       10: {'label': '10', 'style': {'font-size': slider_font_size}}})
                         
                            ],
                            style = {"width" : slider_width,
                                     "padding-top" : "5px",
                                     "padding-bottom" : "10px",
                                     "padding-left" : slider_padding,
                                     "padding-right" : slider_padding,
                                     "background-color" : "white"})
                                     #"border-style" : "solid",
                                     #"border-color" : "#CCCCCC",
                                     #"border-width" : "1px",
                                     #"border-radius" : "5px"
               
                    ],
                    style = {"width" : question_width,
                             "background-color" : "white",
                             "padding-bottom" : "20px",
                             "padding-top" : "0px"}),
                
                ], 
                style = {"display" : "flex", 
                         "align-items": "center", 
                         "justify-content": "center",
                         "padding-bottom" : "8px"}),
           
            html.Div([
                
                    html.P("5. Government should make much greater effort to make incomes equal (0 = much greater)",
                    style = {"font-size" : text_size,
                             "margin-left" : text_left_margin,
                             "width" : "360px"}),
                
                ],
                style = {"width" : question_width,
                         "background-color" : "white",
                         "padding-top" : "5px",
                         "padding-bottom" : "0px"}),
             
         
            html.Div([
                
                html.Div([
                
                    html.Div([
                
                            dcc.Slider(
                                id = "income",
                                value = 5,
                                min = 0,
                                max = 10,
                                marks={0: {'label': '0', 'style': {'font-size': slider_font_size}},
                                       1: {'label': '1', 'style': {'font-size': slider_font_size}},
                                       2: {'label': '2', 'style': {'font-size': slider_font_size}},
                                       3: {'label': '3', 'style': {'font-size': slider_font_size}},
                                       4: {'label': '4', 'style': {'font-size': slider_font_size}},
                                       5: {'label': '5', 'style': {'font-size': slider_font_size}},
                                       6: {'label': '6', 'style': {'font-size': slider_font_size}},
                                       7: {'label': '7', 'style': {'font-size': slider_font_size}},
                                       8: {'label': '8', 'style': {'font-size': slider_font_size}},
                                       9: {'label': '9', 'style': {'font-size': slider_font_size}},
                                       10: {'label': '10', 'style': {'font-size': slider_font_size}}})
                         
                            ],
                            style = {"width" : slider_width,
                                     "padding-top" : "5px",
                                     "padding-bottom" : "10px",
                                     "padding-left" : slider_padding,
                                     "padding-right" : slider_padding,
                                     "background-color" : "white"})
                                     #"border-style" : "solid",
                                     #"border-color" : "#CCCCCC",
                                     #"border-width" : "1px",
                                     #"border-radius" : "5px"
               
                    ],
                    style = {"width" : question_width,
                             "background-color" : "white",
                             "padding-bottom" : "20px",
                             "padding-top" : "0px"}),
                
                ], 
                style = {"display" : "flex", 
                         "align-items": "center", 
                         "justify-content": "center",
                         "padding-bottom" : "8px"}),
            
            html.Div([
                
                 html.Div([
                
                    html.P("6. Agree/disagree: there is currently one law for the rich and one for the poor",
                    style = {"font-size" : text_size,
                             "width" : "360px",
                             "padding-bottom" : "0px"}),
                
                    dcc.Dropdown(
                        id='rich-rule',
                        value = 4,
                        options=[
                            {'label': "Strongly disagree", 'value': 0},
                            {'label': "Disagree", 'value': 1},
                            {'label': "Neither agree nor disagree", 'value': 2},
                            {'label': "Agree", 'value': 3},
                            {'label': "Strongly agree", 'value': 4},
                            {'label': "Don't know", 'value': 5}],
                        clearable = False,
                        style = {'width': "360px",
                                 "font-size" : text_size})
                
                    ],
                     style = {"padding-left" : "29px"}),
                
                ],
                style = {"width" : question_width,
                         "background-color" : "white",
                         "padding-top" : "0px",
                         "padding-bottom" : "15px",
                         "padding-left" : "0px"}),
            
            html.Div([
                
                 html.Div([
                
                    html.P("7. Agree/disagree: If welfare benefits weren’t so generous, people would learn to stand on their own two feet",
                    style = {"font-size" : text_size,
                             "width" : "360px",
                             "padding-bottom" : "0px"}),
                
                    dcc.Dropdown(
                        id='welfare',
                        value = 4,
                        options=[
                            {'label': "Strongly disagree", 'value': 0},
                            {'label': "Disagree", 'value': 1},
                            {'label': "Neither agree nor disagree", 'value': 2},
                            {'label': "Agree", 'value': 3},
                            {'label': "Strongly agree", 'value': 4},
                            {'label': "Don't know", 'value': 5}],
                        clearable = False,
                        style = {'width': "360px",
                                 "font-size" : text_size})
                
                    ],
                     style = {"padding-left" : "29px"}),
                
                ],
                style = {"width" : question_width,
                         "background-color" : "white",
                         "padding-top" : "0px",
                         "padding-bottom" : "15px",
                         "padding-left" : "0px"}),
            
            html.Div([
                
                 html.Div([
                
                    html.P("8. Do you think that trade unions...",
                    style = {"font-size" : text_size,
                             "width" : "360px",
                             "padding-bottom" : "0px"}),
                
                    dcc.Dropdown(
                        id='trade-union-1',
                        value = 1,
                        options=[
                            {'label': "Don't have too much power", 'value': 0},
                            {'label': "Have too much power", 'value': 1},
                            {'label': "Don't know", 'value': 2}],
                        clearable = False,
                        style = {'width': "360px",
                                 "font-size" : text_size})
                
                    ],
                     style = {"padding-left" : "29px"}),
                
                ],
                style = {"width" : question_width,
                         "background-color" : "white",
                         "padding-top" : "0px",
                         "padding-bottom" : "15px",
                         "padding-left" : "0px"}),
            
            html.Div([
                
                 html.Div([
                
                    html.P("9. Agree/disagree: no need for strong trade unions to protect employees' working conditions and wages",
                    style = {"font-size" : text_size,
                             "width" : "360px",
                             "padding-bottom" : "0px"}),
                
                    dcc.Dropdown(
                        id='trade-union-2',
                        value = 4,
                        options=[
                            {'label': "Strongly disagree", 'value': 0},
                            {'label': "Disagree", 'value': 1},
                            {'label': "Neither agree nor disagree", 'value': 2},
                            {'label': "Agree", 'value': 3},
                            {'label': "Strongly agree", 'value': 4},
                            {'label': "Don't know", 'value': 5}],
                        clearable = False,
                        style = {'width': "360px",
                                 "font-size" : text_size})
                
                    ],
                     style = {"padding-left" : "29px"}),
                
                ],
                style = {"width" : question_width,
                         "background-color" : "white",
                         "padding-top" : "0px",
                         "padding-bottom" : "15px",
                         "padding-left" : "0px"}),
            
            html.Div([
                
                 html.Div([
                
                    html.P("10. Should voting system be changed to allow smaller political parties to get a fairer share of MPs",
                    style = {"font-size" : text_size,
                             "width" : "360px",
                             "padding-bottom" : "0px"}),
                
                    dcc.Dropdown(
                        id='voting-system',
                        value = 1,
                        options=[
                            {'label': "Keep it as is", 'value': 0},
                            {'label': "We should change the voting system", 'value': 1},
                            {'label': "Don't know", 'value': 2}],
                        clearable = False,
                        style = {'width': "360px",
                                 "font-size" : text_size})
                
                    ],
                     style = {"padding-left" : "29px"}),
                
                ],
                style = {"width" : question_width,
                         "background-color" : "white",
                         "padding-top" : "0px",
                         "padding-bottom" : "20px",
                         "padding-left" : "0px"}),
            
            
            ], style = {"background-color" : "white",
                        "padding-bottom" : "20px",
                        "padding-top" : "25px",
                        "border-radius" : "10px",
                        "border-style" : "solid",
                        "border-color" : "white",
                        "border-width" : "1px"}),
        
        html.Div([
            
            html.Div([
            
                    dcc.Graph(
                        id = "party-preference",
                        style = {"padding-left" : "10px",
                                 "padding-right" : "10px",
                                 "padding-top" : "10px",
                                 "padding-bottom" : "10px"})
            
                ], style = {"background-color" : "white",
                            "height" : "520px",
                            "width" : "100%",
                            "margin-left" : "20px",
                            "margin-right" : "20px",
                            "border-radius" : "10px",
                            "border-style" : "solid",
                            "border-color" : "white",
                            "border-width" : "1px"}),                
            
            html.Div([
                            
                    html.Div([
                
                        html.P(
                            id = "predictions-text",
                            style = {"padding-top" : "25px",
                                     "padding-bottom" : "20px",
                                     "margin-left" : "40px",
                                     "margin-right" : "40px",
                                     'textAlign' : 'center',
                                     "font-size" : "20px",
                                     "font-weight" : "bold"})
                
                    ], id = "predictions-text-background"),                   
    
                ], 
                style = {"background-color" : "white",
                         "width" : "100%",
                         "margin-top" : "10px",
                         "margin-left" : "20px",
                         "margin-right" : "20px",
                         "border-radius" : "10px",
                         "border-style" : "solid",
                         "border-color" : "white",
                         "border-width" : "1px",
                         "display" : "flex", 
                         "align-items": "center", 
                         "justify-content": "center"}),
                            
            ], style = {"background-color" : "#B9D0DF",
                        "height" : "1010px",
                        "width" : "60%"}),
            
        ], style = {"background-color" : "#B9D0DF",
                    "padding-top" : "20px",
                    "padding-bottom" : "30px",
                    "padding-left" : "0px",
                    "display" : "flex",
                    "align-items": "center",
                    "justify-content": "center"}),
                                
    ], 
    style = {"background-color" : "#B9D0DF",
             "padding-bottom" : "40px"})

                    
@app.callback(
    [dash.dependencies.Output('party-preference', 'figure'),
     dash.dependencies.Output('predictions-text', 'children'),
     dash.dependencies.Output('predictions-text', 'style'),
     dash.dependencies.Output('predictions-text-background', 'style')],
    [dash.dependencies.Input('age', 'value'),
     dash.dependencies.Input('region', 'value'),
     dash.dependencies.Input('trust', 'value'),
     dash.dependencies.Input('left-right', 'value'),
     dash.dependencies.Input('income', 'value'),
     dash.dependencies.Input('rich-rule', 'value'),
     dash.dependencies.Input('welfare', 'value'),
     dash.dependencies.Input('trade-union-1', 'value'),
     dash.dependencies.Input('trade-union-2', 'value'),
     dash.dependencies.Input('voting-system', 'value')])
def update_graph(age_value, region_value, trust_value,
                 left_right_value, income_value, rich_rule_value,
                 welfare_value, trade_union_1_value, trade_union_2_value,
                 voting_system_value):
    
    
    ## Build blank prediction vector
    x_columns = ["e01", "region_east midlands", "region_east of england",
       "region_london", "region_north east", "region_north west",
       "region_scotland", "region_south east", "region_south west",
       "region_wales", "region_west midlands",
       "region_yorkshire and the humber", "Age", "l09", "n03", "w11",
       "w11 - don't know", "w15_1", "w15_1 - don't know", "f01_2",
       "f01_2 - don't know", "f01_5", "f01_5 - don't know", "v01",
       "v01 - don't know"]
    
    x_predict = pd.DataFrame(columns = x_columns, index = [0], data = 0)
    
    ## Create scaled age value
    age_scaled = (age_value - scaling["Age"][0]) / scaling["Age"][1]
    
    ## Update vector
    x_predict.loc[0,"Age"] = age_scaled
    
    ## Find column that includes region value
    region_column = x_predict.columns[x_predict.columns.str.contains(region_value)][0]

    ## Put one in that column
    x_predict.loc[0,region_column] = 1
    
    ## Create scaled trust value
    trust_scaled = (trust_value - scaling["n03"][0]) / scaling["n03"][1]

    ## Update vector
    x_predict.loc[0,"n03"] = trust_scaled
    
    ## Create scaled trust value
    left_right_scaled = (left_right_value - scaling["e01"][0]) / scaling["e01"][1]

    ## Update vector
    x_predict.loc[0,"e01"] = left_right_scaled
    
    ## Create scaled income value
    income_scaled = (income_value - scaling["l09"][0]) / scaling["l09"][1]

    ## Update vector
    x_predict.loc[0,"l09"] = income_scaled
    
    ## If don't know is selected then use the median for 'f01_2' and 1 for 'f01_2 - don't know' and scale both
    ## Otherwise use the value selected for 'f01_2' and 0 for 'f01_2 - don't know' and scale both
    if rich_rule_value == 5:
        rich_rule_scaled = (medians["f01_2"][0] - scaling["f01_2"][0]) / scaling["f01_2"][1]
        x_predict.loc[0,"f01_2"] = rich_rule_scaled
        rich_rule_dk_scaled = (1 - scaling["f01_2 - don't know"][0]) / scaling["f01_2 - don't know"][1]
        x_predict.loc[0,"f01_2 - don't know"] = rich_rule_dk_scaled    
    else:
        rich_rule_scaled = (rich_rule_value - scaling["f01_2"][0]) / scaling["f01_2"][1]
        x_predict.loc[0,"f01_2"] = rich_rule_scaled
        rich_rule_dk_scaled = (0 - scaling["f01_2 - don't know"][0]) / scaling["f01_2 - don't know"][1]
        x_predict.loc[0,"f01_2 - don't know"] = rich_rule_dk_scaled
    
    ## If don't know is selected then use the median for 'w15_1' and 1 for 'w15_1 - don't know' and scale both
    ## Otherwise use the value selected for 'w15_1' and 0 for 'w15_1 - don't know' and scale both
    if welfare_value == 5:
        welfare_scaled = (medians["w15_1"][0] - scaling["w15_1"][0]) / scaling["w15_1"][1]
        x_predict.loc[0,"w15_1"] = welfare_scaled
        welfare_dk_scaled = (1 - scaling["w15_1 - don't know"][0]) / scaling["w15_1 - don't know"][1]
        x_predict.loc[0,"w15_1 - don't know"] = welfare_dk_scaled
    else:
        welfare_scaled = (welfare_value - scaling["w15_1"][0]) / scaling["w15_1"][1]
        x_predict.loc[0,"w15_1"] = welfare_scaled
        welfare_dk_scaled = (0 - scaling["w15_1 - don't know"][0]) / scaling["w15_1 - don't know"][1]
        x_predict.loc[0,"w15_1 - don't know"] = welfare_dk_scaled
    
    ## If don't know is selected then use the median for 'w11' and 1 for 'w11 - don't know' and scale both
    ## Otherwise use the value selected for 'w11' and 0 for 'w11 - don't know' and scale both
    if trade_union_1_value == 2:
        trade_union_1_scaled = (medians["w11"][0] - scaling["w11"][0]) / scaling["w11"][1]
        x_predict.loc[0,"w11"] = trade_union_1_scaled
        trade_union_1_dk_scaled = (1 - scaling["w11 - don't know"][0]) / scaling["w11 - don't know"][1]
        x_predict.loc[0,"w11 - don't know"] = trade_union_1_dk_scaled
    else:
        trade_union_1_scaled = (trade_union_1_value - scaling["w11"][0]) / scaling["w11"][1]
        x_predict.loc[0,"w11"] = trade_union_1_scaled
        trade_union_1_dk_scaled = (0 - scaling["w11 - don't know"][0]) / scaling["w11 - don't know"][1]
        x_predict.loc[0,"w11 - don't know"] = trade_union_1_dk_scaled    

    ## If don't know is selected then use the median for 'f01_5' and 1 for 'f01_5 - don't know' and scale both
    ## Otherwise use the value selected for 'f01_5' and 0 for 'f01_5 - don't know' and scale both
    if trade_union_2_value == 5:
        trade_union_2_scaled = (medians["f01_5"][0] - scaling["f01_5"][0]) / scaling["f01_5"][1]
        x_predict.loc[0,"f01_5"] = trade_union_2_scaled
        trade_union_2_dk_scaled = (1 - scaling["f01_5 - don't know"][0]) / scaling["f01_5 - don't know"][1]
        x_predict.loc[0,"f01_5 - don't know"] = trade_union_2_dk_scaled
    else:
        trade_union_2_scaled = (trade_union_2_value - scaling["f01_5"][0]) / scaling["f01_5"][1]
        x_predict.loc[0,"f01_5"] = trade_union_2_scaled
        trade_union_2_dk_scaled = (0 - scaling["f01_5 - don't know"][0]) / scaling["f01_5 - don't know"][1]
        x_predict.loc[0,"f01_5 - don't know"] = trade_union_2_dk_scaled

    ## If don't know is selected then use the median for 'v01' and 1 for 'v01 - don't know' and scale both
    ## Otherwise use the value selected for 'v01' and 0 for 'v01 - don't know' and scale both
    if voting_system_value == 2:
        voting_system_scaled = (medians["v01"][0] - scaling["v01"][0]) / scaling["v01"][1]
        x_predict.loc[0,"v01"] = voting_system_scaled
        voting_system_dk_scaled = (1 - scaling["v01 - don't know"][0]) / scaling["v01 - don't know"][1]
        x_predict.loc[0,"v01 - don't know"] = voting_system_dk_scaled
    else:
        voting_system_scaled = (voting_system_value - scaling["v01"][0]) / scaling["v01"][1]
        x_predict.loc[0,"v01"] = voting_system_scaled
        voting_system_dk_scaled = (0 - scaling["v01 - don't know"][0]) / scaling["v01 - don't know"][1]
        x_predict.loc[0,"v01 - don't know"] = voting_system_dk_scaled
    
    ## Create dataframe of predictions

    parties = ["Conservative Party", "Green Party", "Labour Party", "Liberal Democrats", "Plaid Cymru", 
               "SNP", "UKIP"]
    
    predictions = pd.DataFrame(columns = parties, data = calibrated.predict_proba(x_predict))
    
    predictions_rounded = predictions * 100
    predictions_rounded = predictions_rounded.round(0)
    predictions_rounded = predictions_rounded.astype(int)
    
    ## Redistribute probabilities for SNP/Plaid Cymru proportionally in situations where those parties cannot be voted for (i.e. region is not Scotland/Wales)
    
    ##if region_value != "wales":
        ##predictions["Plaid Cymru"][0] = 0
        ##predictions_sum = predictions.sum().sum()
        ##for i in parties:
            ##predictions[i][0] = predictions[i][0] / predictions_sum
    
    ##if region_value != "scotland":
        ##predictions["SNP"][0] = 0
        ##predictions_sum = predictions.sum().sum()
        ##for i in parties:
            ##predictions[i][0] = predictions[i][0] / predictions_sum
    
    colours = ["#355C7D", "#3F9D5E", "#F67280", "#F2CE3A", "#45925E", "#F7F787", "#C752C7"]
    
    fig = px.bar(x = predictions.columns, 
                 y = predictions_rounded.loc[0,:],
                 text = predictions_rounded.loc[0,:].astype(str) + "%",
                 height = 500,
                 title = "<b>Predicted probability of feeling closest to each party</b>")
    
    fig.update_traces(marker_color = colours, textposition='outside')
    
    fig.update_layout(
    xaxis_title="Party",
    yaxis_title="Probability of feeling closest to this party",
    title_font_size=24,
    title_font_color="#333333",
    title_x=0.5,
    font=dict(
        size=12))
    
    fig.update_yaxes(range=[0,100])
    
    ## Creating the predictions text with figure from the predictions dataframe
    
    predictions_transposed = predictions_rounded.transpose().sort_values(by = 0, ascending = False)
    
    
    prediction_one = str(predictions_transposed[0][0])
    prediction_two = str(predictions_transposed[0][1])
    prediction_three = str(predictions_transposed[0][2])
    
    """if prediction_one.startswith(("8","11","18")):
        prediction_one = "an " + prediction_one
    else:
        prediction_one = "a " + prediction_one
    if prediction_two.startswith(("8","11","18")):
        prediction_two = "an " + prediction_two
    else:
        prediction_two = "a " + prediction_two
    if prediction_three.startswith(("8","11","18")):
        prediction_three = "an " + prediction_three
    else:
        prediction_three = "a " + prediction_three"""
    
    the_parties = ["Conservative Party", "Green Party", "Labour Party", "Liberal Democrats", "SNP"]
    
    if predictions_transposed.index[0] in the_parties:
        party_one = "the " + predictions_transposed.index[0]
    else:
        party_one = predictions_transposed.index[0]
        
    if predictions_transposed.index[1] in the_parties:
        party_two = "the " + predictions_transposed.index[1]
    else:
        party_two = predictions_transposed.index[1]
        
    if predictions_transposed.index[2] in the_parties:
        party_three = "the " + predictions_transposed.index[2]
    else:
        party_three = predictions_transposed.index[2]
    
    ##predictions_text = "Someone with these survey responses would have " + str(prediction_one) + "% probability of feeling closest to " + party_one + ", " + str(prediction_two) + "% probability of feeling closest to " + party_two + ", and " + str(prediction_three) + "% chance of feeling closest to " + party_three
    
    predictions_text = "This person is most likely to feel closest to " + party_one + " (" + prediction_one + "% probability)"
    
    ## Determining box colour
    
    if predictions_transposed.index[0] == "Conservative Party":
        prediction_text_backcolour = "#355C7D"
        prediction_text_colour = "white"
    elif predictions_transposed.index[0] == "Green Party":
        prediction_text_backcolour = "#3F9D5E"
        prediction_text_colour = "white"
    elif predictions_transposed.index[0] == "Labour Party":
        prediction_text_backcolour = "#F67280"
        prediction_text_colour = "white"
    elif predictions_transposed.index[0] == "Liberal Democrats":
        prediction_text_backcolour = "#F2CE3A"
        prediction_text_colour = "#333333"
    elif predictions_transposed.index[0] == "Plaid Cymru":
        prediction_text_backcolour = "#45925E"
        prediction_text_colour = "white"
    elif predictions_transposed.index[0] == "SNP":
        prediction_text_backcolour = "#F7F787"
        prediction_text_colour = "#333333"
    elif predictions_transposed.index[0] == "UKIP":
        prediction_text_backcolour = "#C752C7"
        prediction_text_colour = "white"    
        
    predictions_text_style = {"color" : prediction_text_colour,
                             "padding-top" : "15px",
                             "padding-bottom" : "10px",
                             "margin-left" : "40px",
                             "margin-right" : "40px",
                             'textAlign' : 'center',
                             "font-size" : "30px",
                             "font-weight" : "bold"}
        
    predictions_div_style = {"background-color" : prediction_text_backcolour,
                             "width" : "100%",
                            "margin-top" : "10px",
                            "margin-bottom" : "10px",
                            "margin-left" : "10px",
                            "margin-right" : "10px",
                            "border-radius" : "6px",
                            "border-style" : "solid",
                            "border-color" : prediction_text_backcolour,
                            "border-width" : "1px"}
    
    return fig, predictions_text, predictions_text_style, predictions_div_style

if __name__ == '__main__':
    app.run_server(debug=True)
