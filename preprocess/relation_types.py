# 数据集字段中 ‘schema_linking_matrix’ 是一个 num_toks * (num_tables + num_col_set) 的列表，列表中元素的含义定义在以下的类中

class SchemaLinkingTypes():
    # ======== Schema Linking ========
    # None-Match
    NONE = 0
    # Exact-Match
    Q_T_E = 1  # Query-Table-Exact
    Q_C_E = 2
    # Partial-Match
    Q_T_P = 3
    Q_C_P = 4
    # ======== Schema Content Linking ========
    Q_T_C_E = 5  # Query-Table-Content-Exact
    Q_T_C_P = 6  # Query-Table-COntent-Partial
    Q_C_C_E = 7  # Query-Column-Content-Exact
    Q_C_C_P = 8  # Query-Column-Content-Partial

    @classmethod
    def type_num(cls):
        return 9
