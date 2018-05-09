
# coding: utf-8

# In[ ]:


"""
Name        : Halloween Search - The Efficient Reference Recommendation Engine - Training the model part
Description : Extract and store the data and finally exports all the data in a pickle file
Purpose     : Created for the Capgemini Global Data Science Challenge IV
Author:     : Arka Prava Bandyopadhyay
Team        : Team_Halloween
Date:       : 21/03/2018
Version:    : 3.33
"""


# In[34]:


##########################################################################
###########       Convert ppt files to xml file  #########################
##########################################################################
import os
# ***IMPORTANT*** Please put the correct path below where the ppts are saved ***IMPORTANT***
path="PLEASE GIVE PATH"

content= '''#!/bin/bash

path='''+str('"'+path+'"')+'''
libreoffice --headless --invisible --convert-to pptx "$path"/*.ppt --outdir "$path"
FILES="$path"/*.pptx
cd "$path"
mkdir -p data_txt
for f in $FILES
do
base_name=$(basename "$f")
filename="${base_name%.*}"
unzip -qc "$f" ppt/slides/slide*.xml   | grep -oP '(?<=\<a:t\>).*?(?=\</a:t\>)' > "$path"/data_txt/"$filename".txt
done
'''


with open(str(path+'/to_text.sh'),'w') as out:
    out.write(content)
    out.close()
rc = os.system(str('chmod 777 '+path+'/to_text.sh;bash '+path+'/to_text.sh;'))


# In[5]:


import re
import pandas as pd
import os
import glob
import html
def extractText(mypath, df):
        list_files=mypath
        for f in list_files:
            text_file = open(f, encoding="utf8")
            text_from_txt = text_file.read()
            text_file.close()
            #put text in lowercase
            text_lower = str(text_from_txt).lower()

            #Extract paragraphs from text 
            client_name = f.split('data_txt/')[1].split('.')[0].split('_')[0]
            file_name=f.split('data_txt/')[1].split('.')[0]
            technology_name= re.split('\d_',file_name)[-1]
            full_text=text_from_txt
            vars=[file_name,client_name,technology_name,full_text]
            df.loc[len(df.index)] = vars
        return df

    
def regexp(word1, word2, text):
    found =""
    m=re.search(word1+"(.*)"+word2, text, re.DOTALL)
    if m:
        found = m.group(1)
        #print(found)
    else:
        m=re.search(word1+"(.*)", text, re.DOTALL)
        if m:
            found = m.group(1)
        #else:
            #print('not found')
    return found


def CleanClientName(text):
        if '_' in list(text):
            text=text.split('_')
            text= ' '.join(text)
        if '&' in list(text):
            text=text.split('&')
            text= ' '.join(text)
        text = re.sub("([a-z])([A-Z])","\g<1> \g<2>",text)
        return text
mypath=glob.glob(os.path.join(str(path+'/data_txt/'),"*.txt"))          
columns=['file_name','Client_name', 'Technology Name',"Full_text"]
df = pd.DataFrame(columns=columns)   
df1=extractText(mypath, df)

df1['Client_name']=df1['Client_name'].apply(CleanClientName)


# In[125]:


#Country Name List
country_name = ['Aruba', 'Afghanistan', 'Angola', 'Anguilla', 'Åland Islands', 'Albania', 'Andorra', 'United Arab Emirates', 'Argentina', 'Armenia', 'American Samoa', 'Antarctica', 'French Southern Territories', 'Antigua and Barbuda', 'Australia', 'Austria', 'Azerbaijan', 'Burundi', 'Belgium', 'Benin', 'Bonaire, Sint Eustatius and Saba', 'Burkina Faso', 'Bangladesh', 'Bulgaria', 'Bahrain', 'Bahamas', 'Bosnia and Herzegovina', 'Saint Barthélemy', 'Belarus', 'Belize', 'Bermuda', 'Bolivia, Plurinational State of', 'Brazil', 'Barbados', 'Brunei Darussalam', 'Bhutan', 'Bouvet Island', 'Botswana', 'Central African Republic', 'Canada', 'Cocos (Keeling) Islands', 'Switzerland', 'Chile', 'China', "Côte d'Ivoire", 'Cameroon', 'Congo, The Democratic Republic of the', 'Congo', 'Cook Islands', 'Colombia', 'Comoros', 'Cabo Verde', 'Costa Rica', 'Cuba', 'Curaçao', 'Christmas Island', 'Cayman Islands', 'Cyprus', 'Czechia', 'Germany', 'Djibouti', 'Dominica', 'Denmark', 'Dominican Republic', 'Algeria', 'Ecuador', 'Egypt', 'Eritrea', 'Western Sahara', 'Spain', 'Estonia', 'Ethiopia', 'Finland', 'Fiji', 'Falkland Islands (Malvinas)', 'France', 'Faroe Islands', 'Micronesia, Federated States of', 'Gabon', 'United Kingdom', 'Georgia', 'Guernsey', 'Ghana', 'Gibraltar', 'Guinea', 'Guadeloupe', 'Gambia', 'Guinea-Bissau', 'Equatorial Guinea', 'Greece', 'Grenada', 'Greenland', 'Guatemala', 'French Guiana', 'Guam', 'Guyana', 'Hong Kong', 'Heard Island and McDonald Islands', 'Honduras', 'Croatia', 'Haiti', 'Hungary', 'Indonesia', 'Isle of Man', 'India', 'British Indian Ocean Territory', 'Ireland', 'Iran, Islamic Republic of', 'Iraq', 'Iceland', 'Israel', 'Italy', 'Jamaica', 'Jersey', 'Jordan', 'Japan', 'Kazakhstan', 'Kenya', 'Kyrgyzstan', 'Cambodia', 'Kiribati', 'Saint Kitts and Nevis', 'Korea, Republic of', 'Kuwait', "Lao People's Democratic Republic", 'Lebanon', 'Liberia', 'Libya', 'Saint Lucia', 'Liechtenstein', 'Sri Lanka', 'Lesotho', 'Lithuania', 'Luxembourg', 'Latvia', 'Macao', 'Saint Martin (French part)', 'Morocco', 'Monaco', 'Moldova, Republic of', 'Madagascar', 'Maldives', 'Mexico', 'Marshall Islands', 'Macedonia, Republic of', 'Mali', 'Malta', 'Myanmar', 'Montenegro', 'Mongolia', 'Northern Mariana Islands', 'Mozambique', 'Mauritania', 'Montserrat', 'Martinique', 'Mauritius', 'Malawi', 'Malaysia', 'Mayotte', 'Namibia', 'New Caledonia', 'Niger', 'Norfolk Island', 'Nigeria', 'Nicaragua', 'Niue', 'Netherlands', 'Norway', 'Nepal', 'Nauru', 'New Zealand', 'Oman', 'Pakistan', 'Panama', 'Pitcairn', 'Peru', 'Philippines', 'Palau', 'Papua New Guinea', 'Poland', 'Puerto Rico', "Korea, Democratic People's Republic of", 'Portugal', 'Paraguay', 'Palestine, State of', 'French Polynesia', 'Qatar', 'Réunion', 'Romania', 'Russian Federation', 'Rwanda', 'Saudi Arabia', 'Sudan', 'Senegal', 'Singapore', 'South Georgia and the South Sandwich Islands', 'Saint Helena, Ascension and Tristan da Cunha', 'Svalbard and Jan Mayen', 'Solomon Islands', 'Sierra Leone', 'El Salvador', 'San Marino', 'Somalia', 'Saint Pierre and Miquelon', 'Serbia', 'South Sudan', 'Sao Tome and Principe', 'Suriname', 'Slovakia', 'Slovenia', 'Sweden', 'Swaziland', 'Sint Maarten (Dutch part)', 'Seychelles', 'Syrian Arab Republic', 'Turks and Caicos Islands', 'Chad', 'Togo', 'Thailand', 'Tajikistan', 'Tokelau', 'Turkmenistan', 'Timor-Leste', 'Tonga', 'Trinidad and Tobago', 'Tunisia', 'Turkey', 'Tuvalu', 'Taiwan, Province of China', 'Tanzania, United Republic of', 'Uganda', 'Ukraine', 'United States Minor Outlying Islands', 'Uruguay', 'United States', 'Uzbekistan', 'Holy See (Vatican City State)', 'Saint Vincent and the Grenadines', 'Venezuela, Bolivarian Republic of', 'Virgin Islands, British', 'Virgin Islands, U.S.', 'Viet Nam', 'Vanuatu', 'Wallis and Futuna', 'Samoa', 'Yemen', 'South Africa', 'Zambia', 'Zimbabwe', 'UK','US']


# In[126]:


#####################################################################
############        Scraping The Country   ##########################
#####################################################################


df1['country'] = ''
short_cn = ['UK','US']
import pycountry
for c,i in enumerate(df1['Full_text']):
    i=regexp(r'Country[&a-zA-Z ]*:','Deal',html.unescape(i).replace('\n',' '))
    for country in pycountry.countries:
        if country.name in i:
            if df1.loc[c,('country')] == '':
                df1.loc[c,('country')]=country.name
            else:
                df1.loc[c,('country')] = df1.loc[c,('country')] + ',' + country.name
    for j in short_cn:
            if j in i:
                df1.loc[c,('country')]=j
                
for c,i in enumerate(df1['country']):
    if i== 'United Kingdom':
        df1.loc[c,('country')]= 'UK'


# In[127]:


####################################################################################
########### Taking the entire file name with the suffix(.pptx or ppt) ##############
####################################################################################

df1['file_name_actual']=''
mypath=glob.glob(os.path.join(str(path+'/data_txt/'),"*"))
#mypath_ppt=glob.glob(os.path.join('C:\\Users\\abandyop\\DS\\Presentations\\',"*"))
string_file_name=''
for i in range(len(mypath_ppt)):
    string_file_name+=mypath_ppt[i].split('\\')[-1]+' | '
# Just extracting the ppt or pptx from the file name and writing it as actual file name
for i in range(len(df1)):
    pos=string_file_name.find(df1.loc[i,('file_name')]+'.') + len(df1.loc[i,('file_name')]+'.')
    df1.loc[i,('file_name_actual')] = df1.loc[i,('file_name')]+'.'+ string_file_name[pos:pos+4]


# In[128]:


###################################################################################
###########  Extracting the year and month for each document ######################
###################################################################################

df1['year']=''
df1['month']=''
for count,i in enumerate(mypath):
    try:
        k=re.findall(r'\d{4,8}',i.split('txt\\')[1].split('.')[0])[0]
        if (len(k)==6) or (len(k)==8):
            df1.loc[count,('year')] = k[0:4]
            df1.loc[count,('month')] =k[4:6]
        if len(k)==4:
            if k[0:2]=='20':
                df1.loc[count,('year')] = k[0:4]
                df1.loc[count,('month')] ='' 
            else:
                df1.loc[count,('year')] = str('20'+k[0:2])
                df1.loc[count,('month')] =k[2:4]           
    except:
        df1.loc[count,('year')] = ''
        df1.loc[count,('month')] =''


# In[129]:


###################################################################################################################
########################                  The Sector Dictionary          ##########################################
###################################################################################################################

sector_dict={'Automotive': ['Automotive','Audi','BMW','Borg','automobile','auto','automobiles'] ,
'Electronics & High Tech': ['HTECH' ,'High','Tech','ISOLA','Isola','Electronics'],
'Government & Public Sector': ['Public', 'Sector','PS','Railways','Maharashtra','Lufthansa','HMRC','Corning',
                               'Amtrak','Rail','Municipal','Anser','Government'],
'Consumer Products & Retail': ['CPRDT','Consumer', 'Products', 'Retail','CPRDT','CPRD','Retails','Bumble','Adidas','Mc',
                               "Lowe's",'Ahold','Unilever','Cargill','Beiersdorf','FMCGGiant','Loves','Nestle','Burberry'],
'Healthcare & Life Sciences':['Bio', 'Pharma' , 'Health', 'Life','Pharmaceutical','Pharma','Pharmacitical','Hospital',
                              'USPharma','Astellas','Healthcare'],
'Financial Services':['Financial','FS'],
'Insurance':['Insurance','Allianz','MAAF'],
'Banking & Capital Markets':['Bank','Banking','Deutsche'],
'Manufacturing & Industrial Products': ['Manufacturing','Ferro','Siemens','Boeing','Meggitt','MALS'],
'Natural Resources':['Energy', 'Utilities','Gas','Net','Electric'],
'Media & Entertainment': ['Media & Entertainment' ,'Media','Movie','Studio','Netflix','Entertainment'],
'Telecoms':['TME','Telecom','Telco','Telefonica','T-Mobile','Cable','Telecoms'],
'End User Computing': ['Hydro','SUEZ','GDF']
          }

           
           
#'Transportation & Distribution':'Others':

###################################################################################################################
########################            Decoding Sector for each Row         ##########################################
###################################################################################################################
df1['Sector'] = ''

for k,i in enumerate(df1['Full_text']):
    i=html.unescape(i)
    for key,value in sector_dict.items():
        for item in value:
            item=item.lower()
            if item in regexp('Sector:','',i).lower().split()[0:4]:
                df1.loc[k,['Sector']] = key
            elif item in regexp('Client Industry:','',i).lower().split()[0:4]:
                df1.loc[k,['Sector']] = key
            elif item in regexp('Industry\n:','',i).lower().split()[0:4]:
                df1.loc[k,['Sector']] = key            
            elif item in df1['Client_name'][k].lower().split():
                df1.loc[k,['Sector']] = key
                
###################################################################################################################
####################    Changing Financial Services to banking and Insurance   ####################################
###################################################################################################################                

values=df1[df1['Sector']=='Financial Services'].index.values        
for count in values:
    i=df1.loc[count,'Full_text']
    if i.lower().find('bank')!=-1:
        df1.loc[count,('Sector')] = 'Banking & Capital Markets'
    elif i.lower().find('insurance')!=-1:
        df1.loc[count,('Sector')] = 'Insurance'
    elif i.lower().find('card')!=-1:
        df1.loc[count,('Sector')] = 'Banking & Capital Markets'
        
# Correcting wrong data input by ppt makers
df1.loc[df1['file_name']=='RBS_201706_HRAnalitics','Sector']='Banking & Capital Markets'
df1.loc[df1['file_name']=='BorgWarner_201308_SupplyChain','country']='Germany'



# In[130]:


#############################################################################################
####################            Technologies Used      ######################################
#############################################################################################
df1['Tech']=''
for count,i in enumerate(df1['Full_text']):
    i=html.unescape(i.replace('\n', ' ').replace('\t',' '))
    m=' '.join(regexp('Technologies:','FTES deployed',i).split(' ')[0:20])
    if(m==''):
        m=' '.join(regexp('Technologies :','FTES deployed',i).split(' ')[0:20])
    if(m==''):
        m=' '.join(regexp('Package:','Alliance',i).split(' ')[0:20])
    m=m.split('Project Contacts')[0]
    m=m.split('FTEs deployed')[0]
    m=m.split('Alliance Partners')[0]  
    m=m.replace('<<Details of technologies>>','')
    m=m.replace('<<Technologies>>','')
    m=m.replace('<<technology>>','')
    df1.loc[count,['Tech']]=m
    #print(m)
    #print(' '.join(regexp('Technologies:','FTES deployed',i).split(' ')[0:20]))
    #print('-'*50)

#TechCorpus

TechCorpus=''
for i in df1['Tech']:
    TechCorpus+=i


# In[131]:


import codecs
import string
import subprocess 
import nltk
from nltk.stem.snowball import EnglishStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import WordNetLemmatizer
from string import digits
from nltk.corpus import stopwords

###########################################
##### Define text cleaning function #######
###########################################
def text_cleaning(text, escape_list=[], stop=[]):
    l=[]
    """
    Text cleaning function:
        Input: 
            -text: a string variable, the text to be cleaned
            -escape_list : words not to transform by the cleaning process (only lowcase transformation is needed)  
            -stop : custom stopwords
        Output:
            -text cleaned and stemmed           
    """
    
    
    """ Get stop word list from package"""
    StopWords = list(set(stopwords.words('english')))
    custom_stop = StopWords + stop
    
    """ Step 1: Parse html entities"""
    text = html.unescape(text)
    text=text.replace('\n',' ').replace('\t',' ').replace('’','')
    
    
    
    """ Step 2: Decode special caracters"""
    text = text.encode('utf8').decode('unicode_escape')
    
 
    """ Step 3: Tokenise text: spliting text elements with the TreeBankWordTokenizer method"""
    tokenizer = TreebankWordTokenizer()
    tokenz=[','.join(tokenizer.tokenize(mot)) if mot  not in escape_list else mot  for mot in text.split()  ]
    
    
    """ Step 4: Drop punctuations """
    tokenz=[re.sub(r'[^\w\s]',' ',mot) if mot  not in escape_list else mot  for mot in tokenz  ]
    tokenz = ' '.join(tokenz).split()
       
    """ Step 5.1: Remove stop words """
    tokenz=([token for token in tokenz if token not in custom_stop])
    
    
    """ Step 5.2: Delete digits from text """
    tokenz=([token for token in tokenz if (  (token.isdigit())==False)  ])  

    """ Step 5.3: Remove digits from tokens"""
    remove_digits = str.maketrans('', '', digits)
    tokenz=[token.translate(remove_digits)  if token not in  escape_list else token for token in tokenz   ]
    
    """ Step 6.1: Lowcase the text"""
    tokenz=([token.lower() for token in tokenz])
    
    """ Step 6.2: Lemmatize the text 
     
'''tokenz=[WordNetLemmatizer().lemmatize(token) if token not in escape_list else token for token in tokenz ]'''"""
    """ Step 6.2: Stem the text """
    tokenz=[EnglishStemmer().stem(token) if token not in escape_list else token for token in tokenz ]

    """ Step 6.3: Drop words with one caratcter and proceed last check for stop words after Stemming"""
    tokenz=[token for token in tokenz if (token not in  custom_stop and len(token)>1) ]

    return ' '.join(tokenz)


# In[132]:


# list of words not to transform by the cleaning process
escape_list = []

# Custom stopwords to remove
stop = ["client", "capgemini", "copyright", "understand", "right", "reserved", "project"]

df1['Full_text']=df1['Full_text'].apply(text_cleaning,args=(escape_list,stop))
###################################################################
####### Cleaning the need or purpose or technology name ###########
###################################################################
def TechnologyNameClean(text):
    if 'iotanalytics' in text.lower().split():
        text = text.replace('iotanalytics','iot analytics')
    if 'saphana' in text.lower().split():
        text = text.replace('saphana','sap hana')
        
    if '_' in list(text):
        text=text.split('_')
        text= ' '.join(text)
    if '&' in list(text):
        text=text.split('&')
        text= ' '.join(text)
    text = re.sub(r"([a-z])([A-Z])","\g<1> \g<2>",text)
    return text
        


df1['Technology Name']= df1['Technology Name'].apply(TechnologyNameClean)


# In[133]:


#Counting Text Length for each ppt
df1['text_length']=''
df1['text_length']=df1['Full_text'].apply(lambda x:len(x))


# In[134]:


##############################################################
##############      Exporting the Model   ####################
##############################################################

df1.drop(df1[df1['file_name']== 'Data Science Challenge IV Kickoff_v3'].index,inplace=True)
# Same file so deleting
df1.drop(df1[df1['file_name']== 'Beiersdorf_201305_SAP_BW2'].index,inplace=True)
df1.drop(df1[df1['file_name']== 'Beiersdorf_201306_SAP_BW2'].index,inplace=True)
df1.drop(df1[df1['file_name']== 'Beiersdorf_201312_InventoryManagement2'].index,inplace=True)
df1.drop(df1[df1['file_name']== 'SiemensBT_201609_InnovativeAnalyticsBasedServicesUsingIoT.pptx'].index,inplace=True)


df1.reset_index(drop=True,inplace=True)
import pickle
df1.to_pickle("trained_df.pkl")

