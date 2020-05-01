%% 初期設定
%学習エポック数
epoch = 1000;
%データ数
data_siz = 800;
% クラス数
class_siz = 4;
% コンポーネント数
conponent_siz = 2;
% 入力ベクトルの長さ
vector_siz = 6;
% 誤差関数用の数値
J=0;
% 識別率
cp = 0;
% 識別率用の変数
cd = [0]*data_siz;
% dJn/dwの計算用変数
vari = zeros(class_siz,conponent_siz,vector_siz,data_siz);
% Δw
deltaw = zeros(class_siz,conponent_siz,vector_siz);
% 学習率
study_rate = 0.001;
% 出力
Yk = zeros(data_siz,class_siz) ;
% 学習データによる出力
tyk = zeros(data_siz,class_siz) ;
% 重み
weights = rand(class_siz,conponent_siz,vector_siz);
% o×wのための配列
ow = zeros(class_siz,conponent_siz,vector_siz);
% 識別データ用の配列
inputdata = zeros(800,6);
% 学習データ用の配列
traindata = zeros(800,6);

error= []*1000;

%% データ入力

filename01 ='dis_sig.csv';
filename02 ='dis_T_sig.csv';
filename03 ='lea_sig.csv';
filename04 ='lea_T_sig.csv';

disdata = csvread(filename01);
distdata= csvread(filename02);
leadata = csvread(filename03);
leatdata= csvread(filename04);

in_siz = size(inputdata);

for i = 1:data_siz
    inputdata(i,:) = [1 disdata(i,:) times(disdata(i,1),disdata(i,1)) times(disdata(i,1),disdata(i,2)) times(disdata(i,2),disdata(i,2))];
    traindata(i,:) = [1 leadata(i,:) times(leadata(i,1),leadata(i,1)) times(leadata(i,1),leadata(i,2)) times(leadata(i,2),leadata(i,2))];
end
%% 学習


trainweights = weights;

for m = 1:epoch
    J = 0;
    [tyk aOkm] = forward(data_siz,class_siz,conponent_siz,vector_siz,trainweights,traindata);
    %損失関数の計算
    
    
    
    for i = 1:data_siz
        for j = 1:class_siz
            for k = 1:conponent_siz
                for l = 1:vector_siz
                    vari(j,k,l,i) = (tyk(i,j)-leatdata(i,j))*aOkm(j,k,i)/tyk(i,j)*traindata(i,l);
                    % dJn/dwを求める式、LLGMN解説スライドp.19をもとに説明するなら
                    % vari(クラス数k、コンポーネント数m、ベクトルの長さh、学習データの数n)の四次元配列になっている
                end
            end
        end
        Jn = -1*sum(dot(leatdata(i,:),log(tyk(i,:))));
        J = J + Jn;
    end
    %Δwを求める一括学習の式
    %Δw = 学習データ数n分の重み更新を足し合わせたdJn/dwと-εを足し合わせている
    deltaw = -1*study_rate*sum(vari,4);
    
    disp(J)
    
    error(m) = J;
    %学習による重みの更新
    trainweights =trainweights + deltaw;
end




%% 実行

[Yk Okm] = forward(data_siz,class_siz,conponent_siz,vector_siz,trainweights,inputdata);

for i = 1:data_siz
    cd(i) = dot(Yk(i,:),distdata(i,:));
    if cd(i) > 0.8
        cp = cp + 1;
    end
end



answer = cp/data_siz;

X = sprintf('識別率は %d',answer);
disp(X)

writematrix(Yk,'output.csv');
error = transpose(error);
writematrix(error,'error.csv');

function [Yk aOkm] = forward(data_siz,class_siz,conponent_siz,vector_siz,weights,inputdata)
    aOkm = zeros(class_siz,conponent_siz,vector_siz,data_siz);
    in_siz = size(inputdata);
    for i =1:data_siz
        for j = 1:vector_siz
            weights(class_siz,conponent_siz,j)= 0;
            ow(:,:,j) = inputdata(i,j)*weights(:,:,j);
        end
        Ikm = sum(ow,3);
        ikm_siz=size(Ikm);
        for k = 1:ikm_siz(1)
            for m = 1:ikm_siz(2)
                Okm(k,m)= exp(Ikm(k,m))/sum(exp(Ikm),'all');
                aOkm(k,m,i) = Okm(k,m);
            end
        end
        Ik = sum(Okm,2);
        Yk(i,:) = Ik;
    end
end








