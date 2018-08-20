L=[4 3 3];   % // Defining the layers: Total of 4 layers, # of nodes are 2, 4, 4, 1 respectively from input to output layer
alpha = 0.2;   % //usually alpha < 0, ranging from 0.1 to 1
target_mse=0.05 % // one of the exit condition
Max_Epoch=200  % // one of the exit condition
Min_Error=Inf
Min_Error_Epoch=-1
epoch=0;       % // 1 epoch => One forward and backward sweep of the net for each training sample 
mse =Inf;      % // initializing the Mean Squared Error with a very large value.
Err=[];
Epo=[];
load x.csv
load y.csv
load kcross.csv
B=cell(length(L)-1,1);  % forming the number of Beta/weight matrix needed in between the layers



    

for i=1:length(L)-1        % Assign uniform random values in [-0.7, 0.7] 
      B{i} =[1.4.*rand(L(i)+1,L(i+1))-0.7];	
end 


%Let us allocate places for Term, T 
T=cell(length(L),1);
for i=1:length(L)
	T{i} =ones (L(i),1);
end


%Let us allocate places for activation, i.e., Z
Z=cell(length(L),1);
for i=1:length(L)-1
	Z{i} =zeros (L(i)+1,1); % it does not matter how do we initialize (with '0' or '1', or whatever,) this is fine!
end
Z{end} =zeros (L(end),1);  % at the final layer there is no Bias unit

%Let us allocate places for error term delta, d
d=cell(length(L),1);
for i=1:length(L)
	d{i} =zeros (L(i),1);
end
			
indices = crossvalind('Kfold',kcross,10); % forming 10 cross validation.    

kfoldnumber = 1;
classificationerror = zeros(10,1);
mseperann = zeros(10,1);

while kfoldnumber <= 10
kfoldnumber;
xxtrain = zeros(135,4);
yytrain = zeros(135,3);
xxtest = zeros(15,4);
yytest = zeros(15,3);
testindex = 1;
trainindex = 1;


for index=1:length(indices)
    if indices(index,1) == kfoldnumber
      xxtest(testindex,:) = x(index,1:4);
      yytest(testindex,:) = y(index,1:3);
      testindex = testindex + 1;
      continue;
    end
    xxtrain(trainindex,:) = x(index,1:4);
    yytrain(trainindex,:) = y(index,1:3);
    trainindex = trainindex + 1;
    
end

while (mse > target_mse) && (epoch < Max_Epoch)   % outer loop with exit conditions
[Nx,P]=size(xxtrain);
[Ny,K]=size(yytrain);  
  CSqErr=0; 		% //Cumulative Sq Err of each Sample; we will take the average after computing Nx_th sample (=> mse)

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

     % // Compute error term delta 'd' for each of the node except the input unit
     % -----------------------------------------------------------------------
     d{end}=(Z{end}-Yk).*Z{end}.*(1-Z{end}); % // delta error term for the output layer
    
       for i=length(L)-1:-1:2 
          d{i}=Z{i}(1:end-1).*(1-Z{i}(1:end-1)).*sum(d{i+1}'*B{i}(1:end-1,:)); % //compute the error term for all the hidden layer (and skip the input layer).
       end              

       % Now we will update the parameters/weights
       for i=1:length(L)-1 
          B{i}(1:end-1,:)=B{i}(1:end-1,:)-alpha.*(Z{i}(1:end-1)*d{i+1}'); 
          B{i}(end,:)=B{i}(end,:)-alpha.*d{i+1}';  			% // update weight connected to the bias unit(or, intercept)	
       end              
       
       
  end  % //end of for loop #1
    classificationerror(kfoldnumber,:) = [CSqErr];
    
    CSqErr= (CSqErr) /(Nx);        % //Average error of N sample after an epoch 
    mse=CSqErr 
    mseperann(kfoldnumber,:) = [mse];
   
    epoch  = epoch+1
    
    Err = [Err mse];
    Epo = [Epo epoch];   


    if mse < Min_Error
	Min_Error=mse;
        Min_Error_Epoch=epoch;
    end 

        
					    	
end % //while_end



 % Min_Error    
 % Min_Error_Epoch  
  
%//The NN Node and structure needs to be saved, i.e. save L
    %L
	
%// Now the predicted weight B with least error should be saved in a file to be loaded and to be used for test set/new prediction

%for i=max(size(B))
 %     B{i}
 %end 
  
  %// Test
  %// X=[5.3 3.7 1.5 0.2;7	3.2	4.7	1.4]
   %// ====== Same (or similar) code as we used before for feed-forward part (see above)
  for j=1:15 		    % for loop #1		
      Z{1} = [xxtest(j,:) 1]';   % Load Inputs with bias=1
      %%% //(Note: desired output here) .....  Yk   = Y(j,:)'; 	  % Load Corresponding Desired or Target output
  
      % // forward propagation 
      % //----------------------
      for i=1:length(L)-1
       	     T{i+1} = B{i}' * Z{i};
            
             if (i+1)<length(L)
               Z{i+1}=[(1./(1+exp(-T{i+1}))) ;1];
             else  
               Z{i+1}=(1./(1+exp(-T{i+1}))); 
             end 
      end  % //end of forward propagation 
       Z{end}
  end
kfoldnumber = kfoldnumber+1;  
end
%===============================================================================================================
%plot epoch versus error graph
%plot (Epo,Err)  % plot based on full epoch

%plot (Epo(1:50),Err(1:50)) 
  
  




