
class EsIndex:
    index_name = None
    url_suffix_attr_name = 'entityUrl'
    url_prefix = 'http://'

    def __init__(self, index_name, url_suffix_attr_name, url_prefix):
        self.index_name = index_name
        self.url_suffix_attr_name = url_suffix_attr_name
        self.url_prefix += url_prefix
