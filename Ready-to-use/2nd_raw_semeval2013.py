from shared.sts_model import STSModel

model = STSModel('2nd_raw_semeval2013')
print('Using model type: {}'.format(model.type))
print('Features: {}'.format({x.method_name: x.args for x in model.features}))
print('-' * 100)
print(model.predict('dnes je pekne', 'dnešné počasie ma potešilo'))
print(model.predict('odroda hrozna zreje', 'zreje odroda hrozna'))
