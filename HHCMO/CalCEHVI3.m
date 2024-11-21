function CEI = CalCEHVI3(RealFirstObj,Obj,Con,eObjIndex,eConIndex,ineObjIndex,ineConIndex,MSEO,MSEC,EIsamplesNum,S,S_S,ref)

    %evaluate EHVI  minimum distance
      NRealFront = size(RealFirstObj,1);
      [NPop,M] = size(Obj);
      % normal distribution sampling
      sigmaO=MSEO.^0.5;   
      EIsamples = zeros(EIsamplesNum,M,NPop);
      for i = 1:NPop
        if all(sigmaO(i,:)~=0) 
            EIsamples(:,eObjIndex,i) = mvnrnd(Obj(i,eObjIndex),diag(sigmaO(i,:)),EIsamplesNum);
            EIsamples(:,ineObjIndex,i) =repmat( Obj(i,ineObjIndex),EIsamplesNum,1);
        end
      end 
      
      Lowb =min(reshape(min(EIsamples,[],1),M,[]),[],2) ;
      Zmin = min([RealFirstObj;Obj;Lowb'],[],1);
      Upb = max(reshape(max(EIsamples,[],1),M,[]),[],2) ;
      Zmax = max([RealFirstObj;Obj;Upb'],[],1); 
      
      %% Normalization

      a=Zmax-Zmin;
      % Normalization      
      Obj = Obj - repmat(Zmin,NPop,1);
      RealFirstObj = RealFirstObj - repmat(Zmin,NRealFront,1);
      Obj = Obj./repmat(a,NPop,1);
      RealFirstObj  = RealFirstObj ./repmat(a,NRealFront,1);
      
      MSEO=MSEO./repmat((a(eObjIndex).^2),size(MSEO,1),1);
      sigmaO=MSEO.^0.5;
      sigmaC=MSEC.^0.5;
      
      
      % identify the normalized MC samples not dominated by the Pareto front  
      nSample = size(S,1);
      R_S  = zeros(NRealFront,nSample);      
      for i = 1 : NRealFront
          x        = sum(repmat(RealFirstObj(i,:),nSample,1)-S<=0,2) == M;  
          R_S(i,x) = 1;  
      end      
      index=(sum(R_S,1) == 0);
      NonDomS=S(index,:);
      nNonDomS=size(NonDomS,1);
      NonDomS_S=S_S(index,index);      
    
       
      % Calcualte hypervolume
      EI=zeros(NPop,1);
      CEI=zeros(NPop,1); 
      for i = 1:NPop
          if  any(sigmaO(i,:)==0) | any(Con(i,ineConIndex)>0)
              EI(i)=0;
          else
              eisamples = EIsamples(:,:,i);
              %normalize
              eisamples = eisamples - repmat(Zmin,EIsamplesNum,1);
              eisamples = eisamples./repmat(a,EIsamplesNum,1);
              
              %identify the normal samples not dominated by the Pareto front 
              R_s  = zeros(NRealFront,EIsamplesNum);      
              for k = 1 : NRealFront
                  x = sum(repmat(RealFirstObj(k,:),EIsamplesNum,1)-eisamples<=0,2) == M | any(eisamples>ref,2);  
                  R_s(i,x) = 1;  
              end      
              index=(sum(R_s,1) == 0);
              eisamples=eisamples(index,:);             

  
              D = pdist2(eisamples,NonDomS,'euclidean');
              [minValues, columnIndices] = min(D, [], 2);
              improvement = sum(NonDomS_S(columnIndices,:),2);
              
              EI(i) = sum(improvement)/EIsamplesNum;

              %%POF
              PF = 1;
              if size(eConIndex,2) ~= 0
                  for j =  size(eConIndex,2)
                      index=eConIndex(j);
                      pf=normcdf(0 ,Con(i,index),sigmaC(i,j));
                      modpf=min(0.8,pf);
                      PF = PF*modpf;
                  end
              end   
              CEI(i)=EI(i)*PF;                            
          end
      end           
  end       
     
