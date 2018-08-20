
L=[4 4 3];   % // Defining the layers: Total of 4 layers, # of nodes are 2, 4, 4, 1 respectively from input to output layer
alpha = 0.2;   % //usually alpha < 0, ranging from 0.1 to 1
target_mse=0.05 % // one of the exit condition
Max_Epoch=2000  % // one of the exit condition
Min_Error=Inf
Min_Error_Epoch=-1
epoch=0;       % // 1 epoch => One forward and backward sweep of the net for each training sample 
mse =Inf;
mse_test =Inf;
cse= Inf;% // initializing the Mean Squared Error with a very large value.
Err_train=[];
Err_test=[];
cse_test =Inf;
cse_train =Inf;
cse_Err_train=[];
cse_Err_test=[];
Epo=[];

[Nx,P]=size(xxtrain); % // Nx = # of sample in X, P= # of feature in X
[Ny,K]=size(yytrain); % // Ny = # of target output in Y, K= # of class for K classes when K>=3 otherwise, K=1 (for Binary case)

% Optional: Since input and output are kept in different files, it is better to verify the loaded sample size/dimensions.
if Nx ~= Ny 
      error ('The input/output sample sizes do not match');
end


% Optional
if L(1) ~= P
       error ('The number of input nodes must be equal to the size of the features')' 
end 

% Optional
if L(end) ~= K
       error ('The number of output node should be equal to K')' 
end 

B=cell(length(L)-1,1);  % forming the number of Beta/weight matrix needed in between the layers
B0= cell(length(L)-1,1);
for i=1:length(L)-1        % Assign uniform random values in [-0.7, 0.7] 
      B{i} = [1.4.*rand(L(i)+1,L(i+1))-0.7];
      BO{i} = zeros(L(i),L(i+1));
end 


%Let us allocate places for Term, T 
T=cell(length(L),1);
for i=1:length(L)
	T{i} =ones (L(i),1);
end


%Let us allocate places for activation, i.e., Z
Z=cell(length(L),1);
for i=1:length(L)-1
	Z{i} =zeros(L(i)+1,1); % it does not matter how do we initialize (with '0' or '1', or whatever,) this is fine!
end
Z{end} =zeros(L(end),1);  % at the final layer there is no Bias unit

%Let us allocate places for error term delta, d
d=cell(length(L),1);
d1=cell(length(L),1);
for i=1:length(L)
	d{i} =zeros (L(i),1);
    d1{i}= zeros(L(i),1);
end


while (mse > target_mse) && (epoch < Max_Epoch)   % outer loop with exit conditions
  
  CSqErr=0; 		% //Cumulative Sq Err of each Sample; we will take the average after computing Nx_th sample (=> mse)
  classification_error =0;
  for j=1:Nx 		    % // for loop #1		
      Z{1} = [xxtrain(j,:) 1]';   % // Load Inputs with bias=1
      Yk   = yytrain(j,:)'; 	    % // Load Corresponding Desired or Target output
  
      % forward propagation 
      % ----------------------
      for i=1:length(L)-1
       	     T{i+1} = B{i}' * Z{i};
            
             if (i+1)<length(L)
               Z{i+1}=[(1./(1+exp(-T{i+1}))) ;1];
             else  
               Z{i+1}=(1./(1+exp(-T{i+1}))); 
             end 
      end  % // end of forward propagation 

         
      CSqErr= CSqErr+sum((Yk-Z{end}).^2);  % // collect sample wise Cumulative Sq Err
      [M,index]=max(Z{end});%//Fining the index position of max value in Z
      if Yk(index)~=1%// and checking with the Yk index poisition to see it is one or not
      classification_error=classification_error+1;
      end
     % // Compute error term delta 'd' for each of the node except the input unit
     % -----------------------------------------------------------------------
     d{end}=(Z{end}-Yk).*Z{end}.*(1-Z{end}); % // delta error term for the output layer
     d1{end}= d{end}+d1{end};
     
       for i=length(L)-1:-1:2 
          d{i}=Z{i}(1:end-1).*(1-Z{i}(1:end-1)).*sum(d{i+1}'*B{i}(1:end-1,:)'); % //compute the error term for all the hidden layer (and skip the input layer).
          d1{i}=d1{i}+d{i};
          BO{i}=BO{i}+(Z{i}(1:end-1)*d{i+1}');
       end                          
       
  end% //end of forward propagation
  % Now we will update the parameters/weights
  for i=1:length(L)-1 
          B{i}(1:end-1,:)=B{i}(1:end-1,:)-(alpha/Nx).*BO{i}; 
          B{i}(end,:)=B{i}(end,:)-(alpha/Nx).*d1{i+1}';  			% // update weight connected to the bias unit(or, intercept)	
  end  
    
    CSqErr= (CSqErr)/(Nx);% //Average error of N sample after an epoch 
    mse_train=CSqErr; 
    classification_error =(classification_error)/(Nx);
    cse_train = classification_error;
    
    Err_train = [Err_train mse_train];
    cse_Err_train=[cse_Err_train cse_train];
    

    if mse < Min_Error
	Min_Error=mse;
    Min_Error_Epoch=epoch;
    end 
    
   %// ================================================================================================================
% ///////////////////////////////////////////////// Test Section ////////////////////////////////////////////////
% // Here I will be using the last B computed to demo test data to classify but you should save and use best B. 
% // Feed forward part will actually be used, assume test points: 1.(0.5,0.3) and 2.(5,4)
% // NOTE: For point (1) the output is expected to be close to zero
% //      For point (2) the output is expected to be close to one. 
    testCSqErr = 0; 
    test_classification_error=0;
for j=1:15 		    % for loop #1		
     
      Z{1} = [xxtest(j,:) 1]';   % Load Inputs with bias=1
      Yk   = yytest(j,:)'; 	  % Load Corresponding Desired or Target output
  
      % // forward propagation 
      % //----------------------
      for i=1:length(L)-1
       	     T{i+1} = B{i}' * Z{i};
            
             if (i+1)<length(L)
               Z{i+1}=[(1./(1+exp(-T{i+1}))) ;1];
             else  
               Z{i+1}=(1./(1+exp(-T{i+1}))); 
             end 
             
      end% //end of forward propagation
   
        testCSqErr= testCSqErr+sum((Yk-Z{end}).^2);
        [N,index]=max(Z{end});%//Fining the index position of max value in Z
      if Yk(index)~=1%// and checking with the Yk index poisition to see it is one or not
      test_classification_error=test_classification_error+1;
      end% // collect sample wise Cumulative Sq Err
    
end
testCSqErr = (testCSqErr) /(15);        % //Average error of 15 sample after an epoch 
   mse_test = testCSqErr;
   Err_test = [Err_test mse_test];
   test_classification_error =(test_classification_error)/(15);
    cse_test = test_classification_error;
    cse_Err_test =[cse_Err_test cse_test];
     
  epoch = epoch+1;
    Epo = [Epo epoch];
        
					    	
end % //while_end
%//The NN Node and structure needs to be saved, i.e. save L
    L
    
	
%// Now the predicted weight B with least error should be saved in a file to be loaded and to be used for test set/new prediction

  for i=max(size(B))
      B{i}
  end 



   trainmse = CSqErr 
   testmse=testCSqErr 
   
   plot (Epo(1:200),Err_train(1:200),'r')
   hold on 
   plot (Epo(1:200),Err_test(1:200),'b')
   figure
   plot(Epo(1:200),cse_Err_train(1:200),'r')
   hold on
   plot(Epo(1:200),cse_Err_test(1:200),'b')
   


