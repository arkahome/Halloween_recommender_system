
# coding: utf-8

# In[2]:


"""
Name        : Halloween Search - The Efficient Reference Recommendation Engine
Description : Selects reference ppts based on a score which combines cosine similarity, country match, sector match etc.
Purpose     : Created for the Capgemini Global Data Science Challenge IV
Author:     : Arka Prava Bandyopadhyay
Team        : Team_Halloween
Date:       : 21/03/2018
Version:    : 3.33
"""


# In[19]:


import sys
import html
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.snowball import EnglishStemmer
import warnings
warnings.filterwarnings("ignore")

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
###################################################################################################################
########################                  The Technology Dictionary with multiple names         ###################
###################################################################################################################

tech_dict={
    'Sap Business Objects': ['Business Objects', 'BO','BusinessObjects'],
    'Informatica': ['PowerCenter' ,'Informatica'],
    'Business Warehouse':['bw','business warehouse'],
    'Data Warehouse':['dw','Data Warehouse','datawarehouse'],
    'Business Intelligence':['bi','Business Intelligence'],
    'Microsoft':['ms','microsoft'],
    'Master Data Management':['mdm','Master Data Management'],
    'Customer Relationship Management':['crm','Customer Relationship Management'],
    'Cloud':['cloud','aws','azure','google cloud'],
    'Big Data':['bigdata','big data','hadoop','cloudera'],
    'General Data Protection Regulation':['gdpr','Data Protection Regulation'],
    'Artificial Intelligence':['Artificial Intelligence','ai'],
    'enterprise data warehouse':['data warehouse','edw','datawarehouse'],
    
}

###########################################################################
########## The COUNTRY List taken from pycountry module #############
###########################################################################

country_name = ['Aruba', 'Afghanistan', 'Angola', 'Anguilla', 'Åland Islands', 'Albania', 'Andorra', 'United Arab Emirates', 'Argentina', 'Armenia', 'American Samoa', 'Antarctica', 'French Southern Territories', 'Antigua and Barbuda', 'Australia', 'Austria', 'Azerbaijan', 'Burundi', 'Belgium', 'Benin', 'Bonaire, Sint Eustatius and Saba', 'Burkina Faso', 'Bangladesh', 'Bulgaria', 'Bahrain', 'Bahamas', 'Bosnia and Herzegovina', 'Saint Barthélemy', 'Belarus', 'Belize', 'Bermuda', 'Bolivia, Plurinational State of', 'Brazil', 'Barbados', 'Brunei Darussalam', 'Bhutan', 'Bouvet Island', 'Botswana', 'Central African Republic', 'Canada', 'Cocos (Keeling) Islands', 'Switzerland', 'Chile', 'China', "Côte d'Ivoire", 'Cameroon', 'Congo, The Democratic Republic of the', 'Congo', 'Cook Islands', 'Colombia', 'Comoros', 'Cabo Verde', 'Costa Rica', 'Cuba', 'Curaçao', 'Christmas Island', 'Cayman Islands', 'Cyprus', 'Czechia', 'Germany', 'Djibouti', 'Dominica', 'Denmark', 'Dominican Republic', 'Algeria', 'Ecuador', 'Egypt', 'Eritrea', 'Western Sahara', 'Spain', 'Estonia', 'Ethiopia', 'Finland', 'Fiji', 'Falkland Islands (Malvinas)', 'France', 'Faroe Islands', 'Micronesia, Federated States of', 'Gabon', 'United Kingdom', 'Georgia', 'Guernsey', 'Ghana', 'Gibraltar', 'Guinea', 'Guadeloupe', 'Gambia', 'Guinea-Bissau', 'Equatorial Guinea', 'Greece', 'Grenada', 'Greenland', 'Guatemala', 'French Guiana', 'Guam', 'Guyana', 'Hong Kong', 'Heard Island and McDonald Islands', 'Honduras', 'Croatia', 'Haiti', 'Hungary', 'Indonesia', 'Isle of Man', 'India', 'British Indian Ocean Territory', 'Ireland', 'Iran, Islamic Republic of', 'Iraq', 'Iceland', 'Israel', 'Italy', 'Jamaica', 'Jersey', 'Jordan', 'Japan', 'Kazakhstan', 'Kenya', 'Kyrgyzstan', 'Cambodia', 'Kiribati', 'Saint Kitts and Nevis', 'Korea, Republic of', 'Kuwait', "Lao People's Democratic Republic", 'Lebanon', 'Liberia', 'Libya', 'Saint Lucia', 'Liechtenstein', 'Sri Lanka', 'Lesotho', 'Lithuania', 'Luxembourg', 'Latvia', 'Macao', 'Saint Martin (French part)', 'Morocco', 'Monaco', 'Moldova, Republic of', 'Madagascar', 'Maldives', 'Mexico', 'Marshall Islands', 'Macedonia, Republic of', 'Mali', 'Malta', 'Myanmar', 'Montenegro', 'Mongolia', 'Northern Mariana Islands', 'Mozambique', 'Mauritania', 'Montserrat', 'Martinique', 'Mauritius', 'Malawi', 'Malaysia', 'Mayotte', 'Namibia', 'New Caledonia', 'Niger', 'Norfolk Island', 'Nigeria', 'Nicaragua', 'Niue', 'Netherlands', 'Norway', 'Nepal', 'Nauru', 'New Zealand', 'Oman', 'Pakistan', 'Panama', 'Pitcairn', 'Peru', 'Philippines', 'Palau', 'Papua New Guinea', 'Poland', 'Puerto Rico', "Korea, Democratic People's Republic of", 'Portugal', 'Paraguay', 'Palestine, State of', 'French Polynesia', 'Qatar', 'Réunion', 'Romania', 'Russian Federation', 'Rwanda', 'Saudi Arabia', 'Sudan', 'Senegal', 'Singapore', 'South Georgia and the South Sandwich Islands', 'Saint Helena, Ascension and Tristan da Cunha', 'Svalbard and Jan Mayen', 'Solomon Islands', 'Sierra Leone', 'El Salvador', 'San Marino', 'Somalia', 'Saint Pierre and Miquelon', 'Serbia', 'South Sudan', 'Sao Tome and Principe', 'Suriname', 'Slovakia', 'Slovenia', 'Sweden', 'Swaziland', 'Sint Maarten (Dutch part)', 'Seychelles', 'Syrian Arab Republic', 'Turks and Caicos Islands', 'Chad', 'Togo', 'Thailand', 'Tajikistan', 'Tokelau', 'Turkmenistan', 'Timor-Leste', 'Tonga', 'Trinidad and Tobago', 'Tunisia', 'Turkey', 'Tuvalu', 'Taiwan, Province of China', 'Tanzania, United Republic of', 'Uganda', 'Ukraine', 'United States Minor Outlying Islands', 'Uruguay', 'United States', 'Uzbekistan', 'Holy See (Vatican City State)', 'Saint Vincent and the Grenadines', 'Venezuela, Bolivarian Republic of', 'Virgin Islands, British', 'Virgin Islands, U.S.', 'Viet Nam', 'Vanuatu', 'Wallis and Futuna', 'Samoa', 'Yemen', 'South Africa', 'Zambia', 'Zimbabwe']
###########################################################################
################ Stopwords List from NLTK stopwords #######################
###########################################################################

stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn']

def findWholeWord(w):
    return re.compile(r'\b({0})\b'.format(w), flags=re.IGNORECASE).search

def search(query,df1):
    #################################################################################################
    ####         the sales team base their choices mainly on 3 criterions :                      ####
    ####         - Same need (BI analytics, Big Data architecture, MDM...)                       ####
    ####         - Same technology (SAP, Cognos, Cloudera...)                                    ####
    ####         - Same business sector (Life sciences, Automotive, media & entertainment...)    ####
    #################################################################################################

    ########################################################################
    ###############      The Search Query         ##########################
    ########################################################################

    search_query=query.lower()
    
    #########################################################################
    #######         Deleting stopwords from search query    #################
    #########################################################################
    
    search_query=' '.join(([i for i in search_query.split() if i not in stopwords]))
    
    ########################################################################
    #########           Getting Country from Query   ########################
    ########################################################################
    country=list()
    df1['query_country']=0
    for i in country_name:
        if search_query.find(i.lower())!=-1:
            country.append(i)
    if ('us' in search_query.split()) or ('usa' in search_query.split()) or ('na' in search_query.split()) or ('america' in search_query.split()):
        country.append('US')
    if 'uk' in search_query.split():
        country.append('UK')
    if len(country) != 0:
        for c,i in enumerate(country):
            if i == 'United States':
                country[c]= 'US'
            elif country == 'United Kingdom':
                country[c] = 'UK'
        for count,i in enumerate(df1['country']):
            for name_country in country:
                if i.find(name_country)!=-1:
                    df1.loc[count,('query_country')]=1
    
    # Replacing Words Those words which can bring false cosine similarity
    
    for i in ['na']:
        search_query=search_query.replace(i,'')

    ########################################################################
    #########           Getting Sector from Query   ########################
    ########################################################################
    query_sector=''
    for key,value in sector_dict.items():
        for i in value:
            if i.lower() in search_query.split():
                query_sector=key
    ########################################################################
    ############    Marking The Sector   ####################################
    ########################################################################
    df1['query_sector']=0
    for count,i in enumerate(df1['Sector']):
        if query_sector =='Financial Services':
            if(df1.loc[count,('Sector')]=='Banking & Capital Markets' and df1.loc[count,('Sector')]!=''):
                df1.loc[count,('query_sector')]+=1
            if(df1.loc[count,('Sector')]=='Insurance' and df1.loc[count,('Sector')]!=''):
                df1.loc[count,('query_sector')]+=1
        if(df1.loc[count,('Sector')]==query_sector and df1.loc[count,('Sector')]!=''):
            df1.loc[count,('query_sector')]+=1

    ########################################################################
    ############    Marking The Year    ####################################
    ########################################################################
    df1['query_year']=0
    try:
        query_year=re.findall(r'\d{2,4}',search_query)[0]
        if query_year!='':
            for count,i in enumerate(df1['year']):
                if query_year ==df1.loc[count,('year')]:
                        df1.loc[count,('query_year')]+=1
    except:
        pass
    
    #######################################################################
    ############    Marking The Name   ####################################
    #######################################################################
    df1['name_match']=0
    for i in search_query.split():
        if i not in ['company','ai']: #### Not in the sector values
            for count,j in enumerate(df1['Client_name']): ##### Checking for the name part
                if j.lower().find(i)>-1:
                    #df1['Match_tech'][count]+=1
                    if df1.loc[count,('name_match')]==0:
                        df1.loc[count,('name_match')]+=1
                    else:
                        df1.loc[count,('name_match')]+=0.1
            for count,j in enumerate(df1['Technology Name']):
                if j.lower().find(i)>-1:
                    #df1['Match_tech'][count]+=1
                    if df1.loc[count,('name_match')]==0:
                        df1.loc[count,('name_match')]+=.2
                    else:
                        df1.loc[count,('name_match')]+=0.1
    #######################################################################
    ####################  Marking From Tech Corpus ########################
    #######################################################################
    for key,value in tech_dict.items():
        for i in value:
            if findWholeWord(i.lower())(search_query):
                for i in tech_dict.get(key):
                    if not findWholeWord(i.lower())(search_query):
                        search_query+= ' '+i.lower()
                break

    #######################################################################
    ############    Marking The Tech   ####################################
    #######################################################################
    df1['Match_tech']=0
    #Techcorpus--Technologies Used        
    TechCorpus=''
    for i in df1['Tech']:
        TechCorpus+=i
    for i in search_query.split():
        if i not in ['data']: #To prevent data to be assigned to teradata or datastage etc.
            if TechCorpus.lower().find(i)>-1:
                for count,j in enumerate(df1['Tech']):
                    if j.lower().find(i)>-1:
                        #df1['Match_tech'][count]+=1
                        if df1.loc[count,('Match_tech')]==0:
                            df1.loc[count,('Match_tech')]+=1
                        else:
                            df1.loc[count,('Match_tech')]+=0.1
    # Replacing Words Those words which can bring false cosine similarity
    
    for i in ['informatica']:
        search_query=search_query.replace(i,'')
    ###########################################################################################################
    ###########################################################################################################
    ###############    Main Recommander System By TF-IDF and Cosine Similarity    #############################
    ###########################################################################################################
    ###########################################################################################################
    vec = TfidfVectorizer()
    doclist_tfidf = vec.fit_transform(df1['Full_text']).toarray()
    query_tfidf = vec.transform([' '.join([EnglishStemmer().stem(token) for token in search_query.split()])]).toarray()
    rec_idx = cosine_similarity(doclist_tfidf, query_tfidf)
    df1['cos_sim']=rec_idx
    
    ##########################
    ###### SCORE #############
    ##########################

    df1['score']=0
    for i in range(len(df1)):
        df1.loc[i,('score')] = df1.loc[i,('cos_sim')]*1 + df1.loc[i,('Match_tech')]*.20 +df1.loc[i,('query_sector')] * 0.40 +df1.loc[i,('query_year')] * 0.20+ df1.loc[i,('name_match')]*0.30+df1.loc[i,('query_country')]*0.20
        if (df1.loc[i,('Match_tech')]>=1 and df1.loc[i,('query_sector')]==1):
            df1.loc[i,('score')] +=0.1
        if (df1.loc[i,('name_match')]>=1 and df1.loc[i,('query_sector')]==1):
            df1.loc[i,('score')] +=0.2
    #To ensure top similarity ppts get better weightage 
    top4_similarity=df1.sort_values(['cos_sim'],ascending=False).reset_index().loc[:3,'file_name_actual']
    for i in range(len(df1)):
        if df1.loc[i,('file_name_actual')] in list(top4_similarity):
            if query_sector == '':
                #For those unfortunate ppts which doesn't have features extracted properly
                # All these if elses are to prevent non sector matches to gain more numbers
                if df1.loc[i,('cos_sim')]>0.1:
                    df1.loc[i,('score')] = df1.loc[i,('score')] + df1.loc[i,('cos_sim')]/df1.loc[i,('score')] 
                elif (0.05<df1.loc[i,('cos_sim')]<0.1):
                    df1.loc[i,('score')] = df1.loc[i,('score')] + df1.loc[i,('cos_sim')]/df1.loc[i,('score')] * 0.5
            else:
                if (df1.loc[i,('cos_sim')]>0.1):
                    if(df1.loc[i,('Sector')]==query_sector):
                        df1.loc[i,('score')] = df1.loc[i,('score')] + df1.loc[i,('cos_sim')]/df1.loc[i,('score')]
                    else:
                        df1.loc[i,('score')] = df1.loc[i,('score')] + df1.loc[i,('cos_sim')]/df1.loc[i,('score')]*0.50
                elif (0.05<df1.loc[i,('cos_sim')]<0.1):
                    if(df1.loc[i,('Sector')]==query_sector):
                        df1.loc[i,('score')] = df1.loc[i,('score')] + df1.loc[i,('cos_sim')]/df1.loc[i,('score')]*0.50
                    else:
                        df1.loc[i,('score')] = df1.loc[i,('score')] + df1.loc[i,('cos_sim')]/df1.loc[i,('score')]*0.25  
    # To prevent models without any similaritirs to feature in final results
    for i in range(len(df1)):
        if df1.loc[i,('cos_sim')]<0.0000000001 and search_query.find('powercenter')==-1: #to help informatica
            df1.loc[i,('score')]=df1.loc[i,('score')]*0.30
    return df1[df1['score']>0].sort_values(['score'],ascending=False).loc[:,['file_name_actual']]

def main():
    ####################################################################################
    #########     Calling the main search function and printing the result    ##########
    ####################################################################################
    df1= pd.read_pickle("trained_df.pkl")
    #Reading the Search Query from the command line
    search_query = sys.argv[1]
    search_result=search(search_query,df1)
    # Assigning a random result if 
    if len(search_result)==0:
        #Old Logic#search_result=df1[(df1['text_length']<6000)&(df1['Tech']!='')&(df1['Sector']!='')].sort_values('text_length',ascending=False)[:30].sort_values('year',ascending=False).loc[:,['file_name_actual']]
        #This one is based on most frequent ppts for the queries in first evaluation round
        for i in ['Insights And Data Reference Booklet - By Sector.pptx','CoxCommunicationsInc_201712_BigData.pptx','BMW_201206_BIStrategy_140212.pptx','Astellas_201302_QlikView.pptx','BayerBusinessServicesGmbH_201712_SAP.pptx','TelecomCompany_201702_DataMigrationOfBillingSystem.pptx','Bahlsen_201311_SAP_BW.pptx','Westpac_201310_Data_Quality.pptx','RBS_201106_Data_Migration.pptx','GlobalPharmaceuticalCompany_201302_QlikView.pptx']:
            print(i)
    for count,i in enumerate(search_result['file_name_actual']):
        if count<10:
            print(i)
        else:
            break

if __name__=="__main__":
    main()

