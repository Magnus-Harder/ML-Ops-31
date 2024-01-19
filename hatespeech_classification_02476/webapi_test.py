import requests
response = requests.get('https://deploy-vout6xuvda-uc.a.run.app')
print(response.status_code)
print(response.json())

response = requests.get('https://deploy-vout6xuvda-uc.a.run.app/predict', params={'input_data':"I really like the flowers you bought"})
print(response.status_code)
print(response.json())

response = requests.get('https://deploy-vout6xuvda-uc.a.run.app/predict', params={'input_data':"just by being able to tweet this insufferable bullshit proves trump a nazi you vagina"})
print(response.status_code)
print(response.json())


# https://deploy-vout6xuvda-uc.a.run.app/predict/?input_data=just by being able to tweet this insufferable bullshit proves trump a nazi you vagina


