from rknn3lite.api.rknn3_types import RKNN3QueryCmd

# 查询输入输出数量
io_num = rknn.rknn3_query(RKNN3QueryCmd.RKNN3_QUERY_IN_OUT_NUM)
print(io_num.n_input, io_num.n_output)

# 查询第 0 个输入张量属性
attr = rknn.rknn3_query(RKNN3QueryCmd.RKNN3_QUERY_INPUT_ATTR, index=0)
print(attr.name, attr.n_dims)

# 查询 SDK 版本
ver = rknn.rknn3_query(RKNN3QueryCmd.RKNN3_QUERY_SDK_VERSION)
print(ver.api_version)
