class AttributeHashmap(dict):
    def __init__(self, *args, **kwargs):
        super(AttributeHashmap, self).__init__(*args, **kwargs)
        self.__dict__ = self
