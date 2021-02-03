import torch
import torch.optim as optim
from KD_Lib.KD import VanillaKD
from torchvision import datasets, transforms
from KD_Lib.models.shallow import Shallow


def main():
    # This part is where you define your datasets, dataloaders, models and optimizers
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "mnist_data",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        ),
        batch_size=32,
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "mnist_data",
            train=False,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        ),
        batch_size=32,
        shuffle=True,
    )

    # two shallow models. number of parameters is halved for the student model.
    # would be interesting to see how much of the accuracy is compromised in the student model.
    teacher_model = Shallow(hidden_size=800)
    student_model = Shallow(hidden_size=200)

    # instantiate optimizers
    teacher_optimizer = optim.SGD(teacher_model.parameters(), 0.01)
    student_optimizer = optim.SGD(student_model.parameters(), 0.01)

    # Now, this is where KD_Lib comes into the picture
    distiller = VanillaKD(teacher_model=teacher_model,
                          student_model=student_model,
                          train_loader=train_loader,
                          val_loader=test_loader,
                          optimizer_teacher=teacher_optimizer,
                          optimizer_student=student_optimizer)

    # here are the code for distillation.
    distiller.train_teacher(epochs=5, plot_losses=True, save_model=True)  # Train the teacher network
    distiller.train_student(epochs=5, plot_losses=True, save_model=True)  # Train the student network
    distiller.evaluate(teacher=False)  # Evaluate the student network
    distiller.get_parameters()  # to compare the number of parameters.


if __name__ == '__main__':
    main()
