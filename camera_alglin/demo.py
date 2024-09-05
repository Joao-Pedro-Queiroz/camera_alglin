import numpy as np
import cv2 as cv

def criar_indices(min_i, max_i, min_j, max_j):
    import itertools
    L = list(itertools.product(range(min_i, max_i), range(min_j, max_j)))
    idx_i = np.array([e[0] for e in L])
    idx_j = np.array([e[1] for e in L])
    idx = np.vstack((idx_i, idx_j))
    return idx

def rotacao(theta):
    # Matriz de rotação 2D para o ângulo theta
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])

def run():
    # Abre a câmera
    cap = cv.VideoCapture(0)

    # Define a largura e altura
    width = 550
    height = 500

    # Ponto central da imagem (em torno do qual faremos a rotação)
    centro_x = width // 2 - 25
    centro_y = height // 2 + 25

    # Matriz de translação para mover o centro da imagem para a origem (0,0)
    T_centro_para_origem = np.array([
        [1, 0, -centro_x], 
        [0, 1, -centro_y], 
        [0, 0, 1]
    ])

    # Matriz de translação inversa (para retornar o centro para sua posição original)
    T_origem_para_centro = np.array([
        [1, 0, centro_x], 
        [0, 1, centro_y], 
        [0, 0, 1]
    ])

    if not cap.isOpened():
        print("Não consegui abrir a câmera!")
        exit()

    # Ângulo inicial de rotação
    theta = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Não consegui capturar frame!")
            break

        # Redimensiona a imagem para trabalhar em uma resolução menor
        frame = cv.resize(frame, (width, height), interpolation=cv.INTER_AREA)

        # Normaliza a imagem para [0,1] e inicializa o buffer da imagem rotacionada
        image = np.array(frame).astype(float) / 255
        image_ = np.zeros_like(image)

        # Atualiza o ângulo de rotação a cada frame
        theta += 0.05  # Ajuste o valor para controlar a velocidade da rotação

        # Calcula a matriz de rotação para o ângulo atual
        R = rotacao(theta)

        # Matriz de transformação composta (translada para o centro, rotaciona, e volta para a posição original)
        Y = T_origem_para_centro @ R @ T_centro_para_origem

        # Gera os índices dos pixels da imagem
        X = criar_indices(0, width, 0, height)
        X = np.vstack((X, np.ones(X.shape[1])))

        # Aplica a transformação
        Xd = Y @ X
        Xd = Xd.astype(int)
        X = X.astype(int)

        # Aplica o clipping nos índices ANTES de acessar a imagem
        Xd[0, :] = np.clip(Xd[0, :], 0, image.shape[0] - 1)  # Limita no eixo vertical
        Xd[1, :] = np.clip(Xd[1, :], 0, image.shape[1] - 1)  # Limita no eixo horizontal
        X[0, :] = np.clip(X[0, :], 0, image.shape[0] - 1)  # Limita os índices originais também
        X[1, :] = np.clip(X[1, :], 0, image.shape[1] - 1)

        # Faz a atribuição dos pixels transformados
        image_[Xd[0, :], Xd[1, :], :] = image[X[0, :], X[1, :], :]

        # Mostra a imagem na tela
        cv.imshow('Minha Imagem!', image_)

        # Sai do loop se 'q' for pressionado
        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()