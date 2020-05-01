%% 初期設定

filename01 ='first.csv';
filename02 ='second.csv';
filename03 ='third.csv';
filename04 ='forth.csv';
filename05 ='tfirst.csv';
filename06 ='tsecond.csv';
filename07 ='tthird.csv';
filename08 ='tforth.csv';

first = readmatrix(filename01);
second= readmatrix(filename02);
third = readmatrix(filename03);
forth = readmatrix(filename04);
t_first = readmatrix(filename05);
t_second= readmatrix(filename06);
t_third = readmatrix(filename07);
t_forth = readmatrix(filename08);

ori = second;
%ori = vertcat(second,third,forth);
orit = t_second;
%orit = vertcat(t_second,t_third,t_forth);
oritrain = third;
%oritrain= vertcat(second,third,forth);
oritraint = t_third;
%oritraint = vertcat(t_second,t_third,t_forth);
in_siz = size(ori);
indata_siz = in_siz(1);
train_siz = size(oritrain);
traindata_siz = train_siz(1);
% 識別データ用の配列
inputdata = zeros(in_siz(1),15);
% 学習データ用の配列
traindata = zeros(train_siz(1),15);



%学習エポック数
epoch = 1000;
% 学習率
study_rate = 0.001;
%データ数

% クラス数
class_siz = 4;
% コンポーネント数
conponent_siz = 2;
% 入力ベクトルの長さ
vector_siz = 15;
% 誤差関数用の数値
J=0;
% 識別率
cp = 0;
% 識別率用の変数
cd = [0]*in_siz(1);
% dJn/dwの計算用変数
vari = zeros(class_siz,conponent_siz,vector_siz,train_siz(1));
% Δw
deltaw = zeros(class_siz,conponent_siz,vector_siz);

% 出力
Yk = zeros(in_siz(1),class_siz) ;
% 学習データによる出力
tyk = zeros(train_siz(1),class_siz) ;
% 重み
weights = rand(class_siz,conponent_siz,vector_siz);
% o×wのための配列
ow = zeros(class_siz,conponent_siz,vector_siz);

%誤差関数の遷移確認用の変数
error= []*1000;

%% データ入力

%データの非線形変換
for i = 1:in_siz(1)
    inputdata(i,:) = [1 ori(i,:) times(ori(i,1),ori(i,1)) times(ori(i,1),ori(i,2)) times(ori(i,1),ori(i,3)) times(ori(i,1),ori(i,4)) times(ori(i,2),ori(i,2)) times(ori(i,2),ori(i,3)) times(ori(i,2),ori(i,4)) times(ori(i,3),ori(i,3)) times(ori(i,3),ori(i,4)) times(ori(i,4),ori(i,4))];
    
end

for i = 1:train_siz(1)
    traindata(i,:) = [1 oritrain(i,:) times(oritrain(i,1),oritrain(i,1)) times(oritrain(i,1),oritrain(i,2)) times(oritrain(i,1),oritrain(i,3)) times(oritrain(i,1),oritrain(i,4)) times(oritrain(i,2),oritrain(i,2)) times(oritrain(i,2),oritrain(i,3)) times(oritrain(i,2),oritrain(i,4)) times(oritrain(i,3),oritrain(i,3)) times(oritrain(i,3),oritrain(i,4)) times(oritrain(i,4),oritrain(i,4))];
    
end

%学習切り替え用の変数(0:一括学習、1:逐次学習)
trainswitch = 0;
%% 一括学習

if trainswitch == 0
    trainweights = weights;
    
    for m = 1:epoch
        
        J = 0; 
        
        [tyk aOkm] = forward(traindata_siz,class_siz,conponent_siz,vector_siz,trainweights,traindata);   
        
        for i = 1:traindata_siz
            for j = 1:class_siz
                for k = 1:conponent_siz
                    for l = 1:vector_siz
                        %dJn/dwを求める式。LLGMN資料p.18,19　式(22),(24)参照
                        vari(j,k,l,i) = (tyk(i,j)-oritraint(i,j))*aOkm(j,k,i)/tyk(i,j)*traindata(i,l);
                        % LLGMN解説スライドp.19をもとに説明するなら
                        % vari(クラス数k、コンポーネント数m、ベクトルの長さh、学習データの数n)の四次元配列になっている
                    end
                end
            end
            Jn = -1*sum(dot(oritraint(i,:),log(tyk(i,:))));
            J=J+Jn;
        end
        %Δwを求める一括学習の式
        %Δw = 学習データ数n分の重み更新を足し合わせたdJn/dwと-εを足し合わせている
        deltaw = -1*study_rate*sum(vari,4);
        
        error(m) = J;
        
        %学習による重みの更新
        trainweights =trainweights + deltaw;
        
    end
    
    X = sprintf('一括学習の誤差関数 %d',J);
    disp(X)
    
end

%% 逐次学習

if trainswitch == 1
    %学習データをランダム化する
    r = randperm(800);
    traindata = traindata(r,:,:,:,:,:,:,:,:,:,:,:,:,:,:);
    t_second = t_second(r,:,:,:,:,:,:,:,:,:,:,:,:,:,:);
    
    trainweights = weights;
    
    for m = 1:epoch
        
        J = 0;
        
        [tyk aOkm] = forward(traindata_siz,class_siz,conponent_siz,vector_siz,trainweights,traindata);
        
        for i = 1:traindata_siz
            for j = 1:class_siz
                for k = 1:conponent_siz
                    for l = 1:in_siz(2)
                        %dJn/dwを求める式。LLGMN資料p.18,19　式(22),(24)参照
                        vari(j,k,l,i) = (tyk(i,j)-oritraint(i,j))*aOkm(j,k,i)/tyk(i,j)*traindata(i,l);
                        % LLGMN解説スライドp.19をもとに説明するなら
                        % vari(クラス数k、コンポーネント数m、ベクトルの長さh、学習データの数n)の四次元配列になっている
                        deltaw = -1*study_rate*vari(:,:,:,i);
                        %学習による重みの更新
                        trainweights =trainweights + deltaw;
                    end
                end
            end
            %誤差関数の計算。LLGMN資料p.18　式(21)参照
            Jn = -1*sum(dot(oritraint(i,:),log(tyk(i,:))));
            J=J+Jn;
        end
        error(m) = J;
    end
    
    X = sprintf('逐次学習の誤差関数 %d',J);
    disp(X)
    
end



%% 実行

[Yk Okm] = forward(indata_siz,class_siz,conponent_siz,vector_siz,trainweights,inputdata);

%識別率の計算出方法について、LLGMNの出力と正解データの内積をとり、内積が閾値を超えれば識別できたと判断しています。
for i = 1:indata_siz
    cd(i) = dot(Yk(i,:),orit(i,:));
    if cd(i) > 0.7
        cp = cp + 1;
    end
end

answer = cp/indata_siz;

if trainswitch == 0
    X = sprintf('一括学習の識別率は %d',answer);
    disp(X)
    writematrix(Yk,'batch_secondthird.csv');
    error = transpose(error);
    writematrix(error,'batch_error.csv');
elseif trainswitch == 1
    X = sprintf('逐次学習の識別率は %d',answer);
    disp(X)
    writematrix(Yk,'online_output.csv');
    error = transpose(error);
    writematrix(error,'online_error.csv');
end



function [Yk aOkm] = forward(data_siz,class_siz,conponent_siz,vector_siz,weights,inputdata)
    aOkm = zeros(class_siz,conponent_siz,vector_siz,data_siz);
    in_siz = size(inputdata);
    for i =1:in_siz(1)
        for j = 1:in_siz(2)
            %重みの一部分を0にする。LLGMN資料p.17参照
            weights(class_siz,conponent_siz,j)= 0;
            %重みと第一層の出力の計算。LLGMN資料p.17 式(15)参照
            ow(:,:,j) = inputdata(i,j)*weights(:,:,j);
        end
        
        Ikm = sum(ow,3);
        ikm_siz=size(Ikm);
        for k = 1:ikm_siz(1)
            for m = 1:ikm_siz(2)
                %指数関数を用いた計算。LLGMN資料p.17 式(16)参照
                Okm(k,m)= exp(Ikm(k,m))/sum(exp(Ikm),'all');
                aOkm(k,m,i) = Okm(k,m);
            end
        end
        %出力の計算
        Ik = sum(Okm,2);
        Yk(i,:) = Ik;
    end
end







