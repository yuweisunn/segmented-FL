# coding: utf-8
import numpy as np

def im2col(input_data, filter_h, filter_w, stride=1, pad=0, constant_values=0):
    """
    input_data : (データ数, チャンネル数, 高さ, 幅)の4次元配列からなる入力データ. 画像データの形式を想定している
    filter_h : フィルターの高さ
    filter_w : フィルターの幅
    stride : ストライド数
    pad : パディングサイズ
    constant_values : パディング処理で埋める際の値
    return : 2次元配列
    """
    
    # 入力データのデータ数, チャンネル数, 高さ, 幅を取得する
    N, C, H, W = input_data.shape 
    
    # 出力データ(畳み込みまたはプーリングの演算後)の形状を計算する
    out_h = (H + 2*pad - filter_h)//stride + 1 # 出力データの高さ(端数は切り捨てる)
    out_w = (W + 2*pad - filter_w)//stride + 1 # 出力データの幅(端数は切り捨てる)

    # パディング処理
    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)],
                             'constant', constant_values=constant_values) # pad=1以上の場合、周囲を0で埋める
    
    # 配列の初期化
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w)) 

    # 配列を並び替える(フィルター内のある1要素に対応する画像中の画素を取り出してcolに代入する)
    for y in range(filter_h):
        """
        フィルターの高さ方向のループ
        """
        y_max = y + stride*out_h
        
        for x in range(filter_w):
            """
            フィルターの幅方向のループ
            """
            x_max = x + stride*out_w
            
            # imgから値を取り出し、colに入れる
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]
            # y:y_max:strideの意味  :  yからy_maxまでの場所をstride刻みで指定している
            # x:x_max:stride の意味  :  xからx_maxまでの場所をstride刻みで指定している

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1) # 軸を入れ替えて、2次元配列(行列)に変換する
    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0, is_maxpooling=False):
    """
    Parameters
    ----------
    col : 2次元配列
    input_shape : 入力データの形状,  (データ数, チャンネル数, 高さ, 幅)の4次元配列
    filter_h : フィルターの高さ
    filter_w : フィルターの幅
    stride : ストライド数
    pad : パディングサイズ
    return : (データ数, チャンネル数, 高さ, 幅)の4次元配列. 画像データの形式を想定している
    -------
    """
    
    # 入力画像(元画像)のデータ数, チャンネル数, 高さ, 幅を取得する
    N, C, H, W = input_shape
    
    # 出力(畳み込みまたはプーリングの演算後)の形状を計算する
    out_h = (H + 2*pad - filter_h)//stride + 1 # 出力画像の高さ(端数は切り捨てる)
    out_w = (W + 2*pad - filter_w)//stride + 1 # 出力画像の幅(端数は切り捨てる)
    
    # 配列の形を変えて、軸を入れ替える
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    # 配列の初期化
    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))  # pad分を大きくとる. stride分も大きくとる
    
    # 配列を並び替える
    for y in range(filter_h):
        """
        フィルターの高さ方向のループ
        """        
        y_max = y + stride*out_h
        for x in range(filter_w):
            """
            フィルターの幅方向のループ
            """            
            x_max = x + stride*out_w
            
            # colから値を取り出し、imgに入れる
            if is_maxpooling:
                img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]
            else:
                img[:, :, y:y_max:stride, x:x_max:stride] = col[:, :, y, x, :, :]
                
    return img[:, :, pad:H + pad, pad:W + pad] # pad分は除いておく(pad分を除いて真ん中だけを取り出す)
