import torch

def train_tiny_micrograd_mlp(inputs: torch.Tensor, targets: torch.Tensor, weights: list[torch.Tensor], biases: list[torch.Tensor], learning_rate: float, steps: int) -> tuple:
    """
    Returns final predictions, final loss, trained weights, trained biases, and pre-update loss history.
    """
    with torch.no_grad():
        L = 0.0
        L_history = []
        outputs = [inputs]
        grads = [inputs]
        w = [wt.clone() for wt in weights]
        b = [bt.clone() for bt in biases]
        if targets.dim() == 1:
            tar = targets.view(-1, 1)
        else:
            tar = targets.clone()
    
        for i in range(len(weights)):
            output = outputs[i]@w[i].T+b[i];
            output = torch.tanh(output)
            if len(outputs) == i + 1: 
                outputs.append(output)
            else:
                outputs[i+1]=output
    
        L = torch.sum((outputs[-1] - tar)*(outputs[-1] - tar))
        if steps == 0: 
            pred = outputs[-1].squeeze(-1) if outputs[-1].dim() > 1 else outputs[-1]
            return (pred, L, w, b, L_history)
        
        for step in range(steps):
            L_history.append(L)
            grads = [None] * (len(w) + 1)
            grads[-1] = 2*(outputs[-1]-tar)*(1-outputs[-1]**2)
            
            for i in reversed(range((len(w)))):
                if i > 0:
                    grads[i]=(grads[i+1]@w[i])*(1-outputs[i]**2)
                weight_grads = grads[i+1].T@outputs[i]
                bias_grads = grads[i+1].sum(dim = 0)
    
                w[i].sub_(learning_rate*weight_grads)
                b[i].sub_(learning_rate*bias_grads)
            
            for i in range(len(w)):
                output = outputs[i]@w[i].T+b[i];
                output = torch.tanh(output)
                if len(outputs) == i + 1: 
                    outputs.append(output)
                else:
                    outputs[i+1]=output
    
            L = torch.sum((outputs[-1] - tar)*(outputs[-1] - tar))
        pred = outputs[-1].squeeze(-1) if outputs[-1].dim() > 1 else outputs[-1]
        return (pred, L, w, b, L_history)
    pass
