from flask import Flask, request, render_template
from numpy import random
import joblib
import os
import pandas as pd

import numpy as np
fet = ['Male',
 'Female',
 'Negroid',
 'Caucasoid',
 'Mongoloid',
 'Australoid',
 'Capoid',
 'Elder Age',
 'Intermediate Age',
 'Young Age',
 'Tropical',
 'Dry',
 'Temperate',
 'Continental',
 'Polar',
 'SkinType_Combination',
 'SkinType_Dry',
 'SkinType_Normal',
 'SkinType_Oily',
 'SkinConcerns_Acne',
 'SkinConcerns_Aging',
 'SkinConcerns_Blackheads',
 'SkinConcerns_Calluses',
 'SkinConcerns_Cellulite',
 'SkinConcerns_Cuticles',
 'SkinConcerns_Dark circles',
 'SkinConcerns_Dullness',
 'SkinConcerns_Pores',
 'SkinConcerns_Puffiness',
 'SkinConcerns_Redness',
 'SkinConcerns_Sensitivity',
 'SkinConcerns_Stretch marks',
 'SkinConcerns_Sun damage',
 'SkinConcerns_Uneven skin tones',
 'SkinTyone_Dark',
 'SkinTyone_Deep',
 'SkinTyone_Ebony',
 'SkinTyone_Fair',
 'SkinTyone_Light',
 'SkinTyone_Medium',
 'SkinTyone_Olive',
 'SkinTyone_Porcelain',
 'SkinTyone_Tan']

acneing = ['Aloe Vera',
 'Althaea Officinalis Root Extract',
 'Arnica Montana Extract',
 'Benzoyl Peroxide',
 'Betaine Salicylate',
 'Chamomile Extract',
 'Commiphora Mukul Resin Extract',
 'Cranberry Extract',
 'Dandelion Extract',
 'Decylene Glycol',
 'Docosahexaenoic Acid',
 'Birch',
 'Coneflower',
 'Farnesol',
 'Fennel',
 'Folic Acid',
 'Genistein',
 'Gluconolactone',
 'Knotweed',
 'Honeysuckle',
 'Garlic',
 'Lactic Acid',
 'Matricaria',
 'Mistletoe',
 'Myrtus Communis',
 'Oat',
 'Oleanolic Acid',
 'Paeonia Lactiflora Root Extract',
 'Palmitoyl Tripeptide-8',
 'Propolis Extract',
 'Pyridoxine Hcl',
 'Raspberry',
 'Rubus Chamaemorus Fruit Extract',
 'Sage',
 'Salicylic Acid',
 'Sigesbeckia Orientalis Extract',
 'Soybean',
 'Sulfur',
 'Magnolia Officinalis Bark Extract',
 'Zinc Sulfate',
 'Lauric Acid',
 'Sigesbeckia Orientalis Extract',
 'Rubus Chamaemorus Fruit Extract',
 'Niacinamide',
 'Phytosphingosine',
 'Potassium Azeloyl Diglycinate/Azelaic Acid',
 'Salix Nigra (Willow) Bark Extract',
 'Zinc Gluconate',
 'Zinc Pca',
 'Reservatrol']

aginging = ['Acetyl Hexapeptide-8',
 'Aloe Barbadensis Extract',
 'Bakuchiol',
 'Caffeine',
 'Centella Asiatica Extract',
 'Colloidal Oatmeal',
 'Copper Tripeptide-1',
 'Haematococcus Pluvialis Extract',
 'Panax Ginseng Root Extract',
 'Acetyl Hexapeptide-1',
 'Adenosine',
 'Ascorbic Acid (Vitamin C/Ascorbyl Glucoside/Magnesium Ascorbyl Phosphate/Tetrahexyldecyl Ascorbate/Ascorbyl Tetraisopalmitate)',
 'Calcium Ketogluconate',
 'Carnosine',
 'Commiphora Mukul Resin Extract',
 'Cranberry',
 'Dipeptide Diaminobutyroyl Benzylamide Diacetate',
 'Eicosapentaenoic Acid',
 'Eucalyptus Globulus',
 'Ferulic Acid',
 'Fucus Vesiculosus Extract',
 'Genistein',
 'Ginkgo Biloba',
 'Gluconolactone',
 'Glycolic Acid',
 'Glycosaminoglycans',
 'Hemp',
 'Green Tea',
 'Hexapeptide-10',
 'Hydrolyzed Extensin',
 'Hydrolyzed Hyaluronic Acid',
 'Hydrolyzed Viola Tricolor Extract',
 'Soy',
 'Lactic Acid',
 'Laminaria',
 'Mandelic Acid',
 'Oxido Reductases',
 'Palmitoyl Pentapeptide-4',
 'Palmitoyl Tripeptide-1',
 'Pea',
 'Phaeodactylum Tricornutum Extract',
 'Polycaprolactone',
 'Polylysine',
 'Proline',
 'Raspberry',
 'Resveratrol',
 'Serenoa Serrulata Fruit Extract',
 'Succinic Acid',
 'RETINOL LIPOSOME',
 'Ubiquinone']


brighting =['Acetyl Glucosamine',
 'Aloe Vera',
 'Apple Cider Vinegar',
 'Arbutin',
 'Ascorbic Acid (Vitamin C/Ascorbyl Glucoside/Magnesium Ascorbyl Phosphate/Tetrahexyldecyl Ascorbate/Ascorbyl Tetraisopalmitate)',
 'Bearberry',
 'Boerhavia Diffusa Root Extract',
 'Chamomile',
 'Chrysin',
 'Citrus Aurantifolia',
 'Cranberry',
 'Cucumis Melo Cantalupensis Fruit Extract',
 'Dimethylmethoxy Chromanol',
 'Ferulic Acid',
 'Gallic Acid',
 'Genistein',
 'Ginkgo Biloba',
 'Glutathione',
 'Glycolic Acid',
 'Hydroquinone',
 'Kojic Acid',
 'Lactic Acid',
 'Licorice',
 'Matricaria',
 'Morus Alba Bark Extract',
 'Pearl Powder',
 'Phenylethyl Resorcinol',
 'Phytic Acid',
 'Potassium Azeloyl Diglycinate (Azelaic Acid)',
 'Resveratrol',
 'Serenoa Serrulata Fruit Extract',
 'Sodium Lactate',
 'Strawberry',
 'Undaria Pinnatifida Extract',
 'Zinc Sulfate',
 'Hexapeptide-2',
 'Hexylresorcinol',
 'Morus Alba Root Extract',
 'Niacinamide',
 'Panax Ginseng root extract',
 'Tetrapeptide-30',
 'Glycine Soja (Soybean) Seed Extrac',
 'Tranexamic Acid',
 'Indian Gooseberry',
 'Reishi',
 'Papaya',
 'Undecylenoyl Phenylalanine',
 'Mulberry',
 'Nasturtium',
 'Undaria Pinnatifida Extract']

# Create Flask object to run
app = Flask(__name__,template_folder= 'templates' )

important_ingredients_acne = ['Benzoyl Peroxide', 'Betaine Salicylate', 'Salicylic Acid', 'Zinc Sulfate', 
                              'Phytosphingosine', 'Potassium Azeloyl Diglycinate/Azelaic Acid', 'Zinc Gluconate', 
                              'Zinc Pca']



important_ingredients_aging = ['Acetyl Hexapeptide-8','Bakuchiol','Centella Asiatica Extract',
                               'Copper Tripeptide-1','Panax Ginseng Root Extract','Acetyl Hexapeptide-1',                                              
                               'Ascorbic Acid (Vitamin C/Ascorbyl Glucoside/Magnesium Ascorbyl Phosphate/Tetrahexyldecyl Ascorbate/Ascorbyl Tetraisopalmitate)', 'Carnosine', 'Genistein', 'Ginkgo Biloba', 'Green Tea', 'Hexapeptide-10', 'Hydrolyzed Hyaluronic Acid','Mandelic Acid','Palmitoyl Pentapeptide-4','Palmitoyl Tripeptide-1','Resveratrol','RETINOL LIPOSOME','Ubiquinone']



important_ingredients_brightning = ['Acetyl Glucosamine','Arbutin',
                                    'Ascorbic Acid (Vitamin C/Ascorbyl Glucoside/Magnesium Ascorbyl Phosphate/Tetrahexyldecyl Ascorbate/Ascorbyl Tetraisopalmitate)',
                                    'Glutathione','Kojic Acid','Lactic Acid','Phenylethyl Resorcinol','Resveratrol','Niacinamide','Tetrapeptide-30','Tranexamic Acid','Undecylenoyl Phenylalanine']




#df_acne_akmal = pd.read_excel('Anti Acne Serum file Akmal (2).xlsx')
#df_aging_akmal = pd.read_excel('Anti aging Serum file Akmal (2).xlsx')
#df_brightning_akmal = pd.read_excel('Brightning Serum file Akmal (2).xlsx')





dir_list_acne = os.listdir('Anti-Acnexx')
dir_list_aging = os.listdir('Anti-agingxx')
dir_list_brightning = os.listdir('Skin Brighteningxx')

  

#df_brightning_akmal['name'] = df_brightning_akmal['name'].str.replace(',','')

def givlis_df(dictionin):
    print(dictionin)
    df = pd.DataFrame(columns =  fet)

    if   (dictionin['Gender'] == 'Male') :

        df['Male'] = [1]
    if   (dictionin['Gender'] == 'Female') :
        df['Female'] = [1]
    if   (dictionin['Age'] == '13-17') or (dictionin['Age'] == '18-24')  or (dictionin['Age'] == '25-34'):
        df['Young Age'] = [1]
        
    
    if   (dictionin['Age'] == '35-44') or (dictionin['Age'] == '45-54')  :  
        df['Intermediate Age'] = [1]
        
    if   (dictionin['Age'] == '55-120') :
        df['Elder Age'] = [1]
        
        
        
    if   (dictionin['Race'] == 'White Skin'):
        df['Australoid'] = [1]  
      
        
    if   (dictionin['Race'] == 'Latino') :
        df['Caucasoid'] = [1]
        
    if  (dictionin['Race'] == 'Middle Eastern') :
        df['Hispanic'] = [1]        
        
    if  (dictionin['Race'] == 'Black Skin') :
        df['Negroid'] = [1] 
      
    if  (dictionin['Race'] == 'South Asian') or (dictionin['Race'] == 'South East Asian') :
        df['Mongoloid'] = [1]     
        
    if  (dictionin['Climate'] == 'Tropical') :
        df['Tropical'] = [1]
        
    if   (dictionin['Climate'] == 'Dry') :
        df['Dry'] = [1]
        
    if   (dictionin['Climate'] == 'Temperate') :
        df['Temperate'] = [1]
        
    if   (dictionin['Climate'] ==  'Continental') :
        df['Continental'] = [1]
        
    if   (dictionin['Climate'] == 'Polar') :
        df['Polar'] = [1]
        
    df['SkinType'+'_'+dictionin['SkinType']] = [1]
    df['SkinTyone'+'_'+dictionin['SkinTyone']] = [1]
    df['SkinConcerns'+'_'+dictionin['SkinConcerns']] = [1]
    df = df.replace(np.NaN,0)
    print(df)
    return df
    
def sorrr(dc):
    dcx = sorted(dc.items(), key=lambda x:x[1],reverse = True)
    return dcx

    
def acne_imp(custtdetails):
    dic_acne_imp = {}
    for i in important_ingredients_acne:
        i = i.replace('/','_')
        if i[0] == ' ':
            i = i[1:]        
        
        for p in dir_list_acne:
            if i in p:
                i = p
                break          
        
        print(p)
        
        xgb = joblib.load(f'Anti-Acnexx/{p}')
        proba = list(xgb.predict_proba(givlis_df(custtdetails))[0])
        pred = xgb.predict(givlis_df(custtdetails))[0]
        #print(i,proba,pred,proba.index(max(proba)))

        dic_acne_imp[i] = proba[1] * 100


    #dic_acne_imp = sorted(dic_acne_imp.items(), key=lambda x:x[1],reverse = True)
    i = None
    
    dic_acne_supp = {} 
   
    for j  in acneing:
        if j not in important_ingredients_acne:
            j = j.replace('/','_')
            if j[0] == ' ':
                j = j[1:]            
            
            for p in dir_list_acne:
                print(p)
                
                if j in p:
                    j = p
                    break              
            xgb = joblib.load(f'Anti-Acnexx/{p}')
            proba = list(xgb.predict_proba(givlis_df(custtdetails))[0])
            pred = xgb.predict(givlis_df(custtdetails))[0]
            #print(i,proba,pred,proba.index(max(proba)))

            dic_acne_supp[j] = proba[1]  * 100
        
    #dic_acne_supp = sorted(dic_acne_supp.items(), key=lambda x:x[1],reverse = True)
        
        
        
    
    z =  [sorrr(dic_acne_imp),'_______________________',sorrr(dic_acne_supp)]
    return z







def aging_imp(custtdetails):
    dic_aging_imp = {}
    for i in important_ingredients_aging:
        
        i = i.replace('/','_')
        if i[0] == ' ':
            i = i[1:]

        
        for p in dir_list_aging:
            if i in p:
                i = p
                break        
        
        
        xgb = joblib.load(f'Anti-agingxx/{p}')
        proba = list(xgb.predict_proba(givlis_df(custtdetails))[0])
        pred = xgb.predict(givlis_df(custtdetails))[0]
        #print(i,proba,pred,proba.index(max(proba)))  
        #print(i,proba,pred,proba.index(max(proba)))

        dic_aging_imp[i] = proba[1] * 100


    #dic_acne_imp = sorted(dic_acne_imp.items(), key=lambda x:x[1],reverse = True)
    i = None
    
    dic_aging_supp = {}    
    for j  in aginging:
        if j not in important_ingredients_aging:
            j = j.replace('/','_')
            if j[0] == ' ':
                j = j[1:]            
            
            for p in dir_list_aging:
                if j in p:
                    j = p
                    break            
            
            
            xgb = joblib.load(f'Anti-agingxx/{p}')
            proba = list(xgb.predict_proba(givlis_df(custtdetails))[0])
            pred = xgb.predict(givlis_df(custtdetails))[0]
            #print(i,proba,pred,proba.index(max(proba)))

            dic_aging_supp[j] = proba[1]  * 100
        
    #dic_acne_supp = sorted(dic_acne_supp.items(), key=lambda x:x[1],reverse = True)
        
        
        
    
    z =  [sorrr(dic_aging_imp),'_______________________',sorrr(dic_aging_supp)]


    return z





def brightning_imp(custtdetails):
    print(custtdetails)
    dic_brightning_imp = {}
    for i in important_ingredients_brightning:
        i = i.replace('/','_')
        print(i)
        if i[0] == ' ':
            i = i[1:]
        for p in dir_list_brightning:
            if i in p:
                print(p)
                i = p
                break
            
        xgb = joblib.load(f'Skin Brighteningxx/{p}')
        proba = list(xgb.predict_proba(givlis_df(custtdetails))[0])
        pred = xgb.predict(givlis_df(custtdetails))[0]
        #print(i,proba,pred,proba.index(max(proba)))  
        #print(i,proba,pred,proba.index(max(proba)))

        dic_brightning_imp[i] = proba[1] * 100


    #dic_acne_imp = sorted(dic_acne_imp.items(), key=lambda x:x[1],reverse = True)
    i = None
    
    dic_brightning_supp = {}    
    for j  in brighting:
        if j not in important_ingredients_brightning:
            j = j.replace('/','_')
            print(j)            
            
            if j[0] == ' ':
                j = j[1:]           
            
            
            for p in dir_list_brightning:
                if j in p:
                    print(p)
                    j = p
                    break            
            xgb = joblib.load(f'Skin Brighteningxx/{p}')
            
            proba = list(xgb.predict_proba(givlis_df(custtdetails))[0])
            pred = xgb.predict(givlis_df(custtdetails))[0]
            #print(i,proba,pred,proba.index(max(proba)))

            dic_brightning_supp[j] = proba[1]  * 100
        
    #dic_acne_supp = sorted(dic_acne_supp.items(), key=lambda x:x[1],reverse = True)
        
        
        
    
    z = [sorrr(dic_brightning_imp),'_______________________',sorrr(dic_brightning_supp)]

    return  z
 
 
 
 
 
 
@app.route('/')
def home():
    return render_template('index.html')



@app.route('/predict',methods=['POST'])
def predict():

   
    SkinConcerns = str(request.form.get('SkinConcerns'))

    Age = str(request.form.get('Age'))

    SkinType = str(request.form.get('SkinType'))

    Gender = str(request.form.get('Gender'))

    SkinTone = str(request.form.get('SkinTyone'))

    Race = str(request.form.get('Race'))
   
    Climate = str(request.form.get('Climate'))  


               

 
    out = {}
   
   

    if (SkinConcerns == 'Acne') or (SkinConcerns == 'Pores')or (SkinConcerns == 'Redness' ):
        custdetails = {'SkinConcerns':SkinConcerns,'Age':Age,'SkinType':SkinType,'SkinTyone':SkinTone,'Gender':Gender,
                  'Race':Race,'Climate':Climate}
 
        out = acne_imp(custdetails)
    if (SkinConcerns == 'Aging') or (SkinConcerns == 'Sun damage') or (SkinConcerns == 'Stretch marks'):
        custdetails = {'SkinConcerns':SkinConcerns,'Age':Age,'SkinType':SkinType,'SkinTyone':SkinTone,'Gender':Gender,
                  'Race':Race,'Climate':Climate}
 
        out = aging_imp(custdetails)          
    if (SkinConcerns == 'Dark circles') or (SkinConcerns == 'Blackheads')   or (SkinConcerns == 'Uneven skin tones')  or (SkinConcerns == 'Dullness'):
        custdetails = {'SkinConcerns':SkinConcerns,'Age':Age,'SkinType':SkinType,'SkinTyone':SkinTone,'Gender':Gender,
                  'Race':Race,'Climate':Climate}
 
        out = brightning_imp(custdetails)     
       
    print(out)

    return render_template('index.html', prediction_text= out)



   
   
if __name__ == "__main__":
    app.run() 
 

   
   


       
