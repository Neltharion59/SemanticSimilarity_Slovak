from shared.sts_model import STSModel

model = STSModel('1st_lemma_all')

print(model.calculate_sts('dnes je pekne', 'dnes je pekne'))
print(model.calculate_sts('odroda hrozna zreje', 'zreje odroda hrozna'))
