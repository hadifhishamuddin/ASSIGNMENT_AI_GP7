%
% Copyright (c) 2015, Yarpiz (www.yarpiz.com)
% All rights reserved. Please read the "license.txt" for license terms.
%
% Project Code: YPAP105
% Project Title: Solving Bin Packing Problem using Genetic Algorithm
% Publisher: Yarpiz (www.yarpiz.com)
% 
% Developer: S. Mostapha Kalami Heris (Member of Yarpiz Team)
% 
% Contact Info: sm.kalami@gmail.com, info@yarpiz.com
%

clc;
clear;
close all;

%% Problem Definition

model = CreateModel();  % Create Bin Packing Model

CostFunction = @(x) BinPackingCost(x, model);  % Objective Function

nVar = 2*model.n-1;     % Number of Decision Variables
VarSize = [1 nVar];     % Decision Variables Matrix Size


%% GA Parameters

MaxIt=1000;         % Maximum Number of Iterations

nPop=40;            % Population Size

pc=0.4;                 % Crossover Percentage
nc=2*round(pc*nPop/2);  % Number of Offsprings (Parents)

pm=0.8;                 % Mutation Percentage
nm=round(pm*nPop);      % Number of Mutants

beta=5;                 % Selection Pressure

%% Initialization

% Create Empty Structure
empty_individual.Position=[];
empty_individual.Cost=[];
empty_individual.Sol=[];


% Create Population Matrix (Array)
pop=repmat(empty_individual,nPop,1);

% Initialize Population
for i=1:nPop
    
    % Initialize Position
    pop(i).Position=randperm(nVar);
    
    % Evaluation
    [pop(i).Cost, pop(i).Sol]=CostFunction(pop(i).Position);
    
end

% Sort Population
Costs=[pop.Cost];
[Costs, SortOrder]=sort(Costs);
pop=pop(SortOrder);

% Update Best Solution Ever Found
BestSol=pop(1);

% Update Worst Cost
WorstCost=max(Costs);

% Array to Hold Best Cost Values
BestCost=zeros(MaxIt,1);


%% GA Main Loop

for it=1:MaxIt
    
    % Calculate Selection Probabilities
    P=exp(-beta*Costs/WorstCost);
    P=P/sum(P);
    
    % Crossover
    popc=repmat(empty_individual,nc/2,2);
    for k=1:nc/2
        
        % Select Parents
        i1=RouletteWheelSelection(P);
        i2=RouletteWheelSelection(P);
        p1=pop(i1);
        p2=pop(i2);
        
        % Apply Crossover
        [popc(k,1).Position, popc(k,2).Position]=PermutationCrossover(p1.Position,p2.Position);
        
        % Evaluate Offsprings
        [popc(k,1).Cost, popc(k,1).Sol]=CostFunction(popc(k,1).Position);
        [popc(k,2).Cost, popc(k,2).Sol]=CostFunction(popc(k,2).Position);
        
    end
    popc=popc(:);
    
    % Mutation
    popm=repmat(empty_individual,nm,1);
    for k=1:nm
        
        % Select Parent Index
        i=randi([1 nPop]);
        
        % Select Parent
        p=pop(i);
        
        % Apply Mutation
        popm(k).Position=PermutationMutate(p.Position);
        
        % Evaluate Mutant
        [popm(k).Cost, popm(k).Sol]=CostFunction(popm(k).Position);
        
    end
    
    % Merge Population
    pop=[pop
         popc
         popm]; %#ok
     
    % Sort Population
    Costs=[pop.Cost];
    [Costs, SortOrder]=sort(Costs);
    pop=pop(SortOrder);
    
    % Truancate Extra Memebrs
    pop=pop(1:nPop);
    Costs=Costs(1:nPop);
    
    % Update Best Solution Ever Found
    BestSol=pop(1);
    
    % Update Worst Cost
    WorstCost=max(WorstCost,max(Costs));
    
    % Update Best Cost Ever Found
    BestCost(it)=BestSol.Cost;
        
    % Show Iteration Information
    disp(['Iteration ' num2str(it) ': Best Cost = ' num2str(BestCost(it))]);
    
end

%% Results

figure;
plot(BestCost,'LineWidth',2);
xlabel('Iteration');
ylabel('Best Cost');

