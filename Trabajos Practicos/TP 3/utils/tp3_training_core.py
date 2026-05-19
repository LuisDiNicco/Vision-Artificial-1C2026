# Este modulo ya no se usa con Keras model.fit().
# Las funciones se mantienen como stubs por compatibilidad de imports.


def train_one_epoch(model, dataset, criterion=None, optimizer=None):
    raise NotImplementedError("Usar model.fit() en vez del loop manual")


def evaluate_one_epoch(model, dataset, criterion=None):
    results = model.evaluate(dataset, verbose=0)
    return results[0], results[1]
