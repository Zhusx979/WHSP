def WHSP_Pooling(feature_maps):
    if feature_maps.size(2) <= 2 and feature_maps.size(3) <= 2:
        
        max_val = torch.max(feature_maps.flatten(start_dim=2), dim=2).values
        return max_val.unsqueeze(-1)  

    top_max = torch.max(feature_maps[:, :, 0, 0:-1], dim=2).values.unsqueeze(2) 
    # print(top_max)
    left_max = torch.max(feature_maps[:, :, 1:, 0], dim=2).values.unsqueeze(2) 
    # print(left_max)
    bottom_max = torch.max(feature_maps[:, :, -1, 1:], dim=2).values.unsqueeze(2) 
    # print(bottom_max)
    right_max = torch.max(feature_maps[:, :, :-1, -1], dim=2).values.unsqueeze(2)
    # print(right_max)

    result = torch.cat((top_max, left_max, bottom_max, right_max), dim=2)
    # print(result)

    inner_matrix = feature_maps[:, :, 1:-1, 1:-1]
    # print(inner_matrix)
    inner_result = WHSP_Pooling(inner_matrix)
    # print(inner_result)
    result = torch.cat((result, inner_result), dim=2)
    # print(result.shape)

    return result

def L2(feature_map):
    l2_vector = torch.norm(feature_map, p=2, dim=2)
    # print(l2_vector)
    return l2_vector

class WHSP_ChannelAttention(nn.Module):
    def __init__(self, in_channels,reduction_ratio=2):
        super(WHSP_ChannelAttention, self).__init__()
        self.in_channels = in_channels
        self.WHSP = WHSP_Pooling  
        self.l2 = L2

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio), 
            nn.Mish(),
            nn.Linear(in_channels // reduction_ratio, in_channels)  
        )

        self.relu1 = nn.Mish()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        tmp = x  

        pooled = self.WHSP(x)  
        l2 = L2(pooled)
        x = self.mlp(l2)
        # x = x + l2
        x = x.view(x.size(0), self.in_channels, 1, 1)  # 形状 [batch, in_planes, 1, 1]

        return self.sigmoid(x) * tmp
