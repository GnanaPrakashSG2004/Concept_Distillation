import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50, resnet18
import wandb
from tqdm import tqdm

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data augmentation and normalization
transform_train = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=False, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=4)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=128, shuffle=False, num_workers=4)

# Initialize models
def create_teacher():
    model = resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)
    return model.to(device)

def create_student():
    model = resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)
    return model.to(device)

# Train teacher model
def train_teacher():
    wandb.init(project="kd-cifar10", name="teacher")
    teacher = create_teacher()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(teacher.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    
    best_acc = 0.0
    for epoch in range(10):
        teacher.train()
        running_loss = 0.0
        for inputs, labels in tqdm(trainloader, desc=f"Teacher Epoch {epoch+1}"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = teacher(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Validation
        teacher.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = teacher(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = 100 * correct / total
        wandb.log({
            "epoch": epoch+1,
            "train_loss": running_loss/len(trainloader),
            "val_acc": val_acc
        })
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(teacher.state_dict(), "teacher.pth")
    
    wandb.finish()
    return teacher

# Knowledge distillation training
def train_distilled_student(teacher):
    wandb.init(project="kd-cifar10", name="distilled-student")
    student = create_student()
    optimizer = optim.SGD(student.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()
    alpha = 0.5  # Distillation loss weight
    
    best_acc = 0.0
    for epoch in range(50):
        student.train()
        running_loss = 0.0
        for inputs, labels in tqdm(trainloader, desc=f"Distill Epoch {epoch+1}"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            with torch.no_grad():
                teacher_logits = teacher(inputs)
            
            student_logits = student(inputs)
            
            loss = alpha * ce_loss(student_logits, labels) + \
                   (1 - alpha) * mse_loss(student_logits, teacher_logits)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Validation
        student.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = student(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = 100 * correct / total
        wandb.log({
            "epoch": epoch+1,
            "train_loss": running_loss/len(trainloader),
            "val_acc": val_acc
        })
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(student.state_dict(), "distilled_student.pth")
    
    wandb.finish()
    return best_acc

# Standard student training
def train_standard_student():
    wandb.init(project="kd-cifar10", name="standard-student")
    student = create_student()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(student.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    
    best_acc = 0.0
    for epoch in range(50):
        student.train()
        running_loss = 0.0
        for inputs, labels in tqdm(trainloader, desc=f"Standard Epoch {epoch+1}"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = student(inputs)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Validation
        student.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = student(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = 100 * correct / total
        wandb.log({
            "epoch": epoch+1,
            "train_loss": running_loss/len(trainloader),
            "val_acc": val_acc
        })
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(student.state_dict(), "standard_student.pth")
    
    wandb.finish()
    return best_acc

if __name__ == "__main__":
    # Train teacher
    teacher = train_teacher()
    
    # Train distilled student
    distilled_acc = train_distilled_student(teacher)
    
    # Train standard student
    standard_acc = train_standard_student()
    
    # Final comparison
    print(f"\n{' Model ':-^30}")
    print(f"Teacher: {torch.load('teacher.pth')['fc.weight'].shape[0]} classes")
    print(f"Distilled Student Accuracy: {distilled_acc:.2f}%")
    print(f"Standard Student Accuracy: {standard_acc:.2f}%")
    print(f"{'':-^30}")