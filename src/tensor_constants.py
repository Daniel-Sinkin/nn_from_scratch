from enum import StrEnum


class Operation(StrEnum):
    """
    ID stands for identity and basically means nothing, for Neurons it
    signifies that we don't have an activation function.
    """

    # Reserved
    NOT_INITIALIZED = "NOT_INITIALIZED"
    ID = "ID"

    # Unary
    EXP = "EXP"
    LOG = "LOG"
    SIN = "SIN"
    COS = "COS"
    RELU = "RELU"
    P_RELU = "P_RELU"
    TANH = "TANH"
    SIGMOID = "SIGMOID"
    SIGMOID_SWISH = "SIGMOID_SWISH"
    POW = "POW"
    TR = "TR"  # Transpose
    MAX = "MAX"
    MIN = "MIN"

    # Reducing
    SUM = "SUM"
    MEAN = "MEAN"

    # Binary
    ADD = "ADD"
    SUB = "SUB"
    MUL = "MUL"
    DIV = "DIV"
    MATMUL = "MATMUL"

    SOFTMAX = "SOFTMAX"
