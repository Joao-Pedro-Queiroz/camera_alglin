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

def escala(s):
    # Matriz de escala uniforme
    return np.array([
        [s, 0, 0],
        [0, s, 0],
        [0, 0, 1]
    ])

def run():
    # Abre a câmera
    cap = cv.VideoCapture(0)

    # Define a largura e altura
    width = 320
    height = 240

    # Ponto central da imagem (em torno do qual faremos a rotação e escala)
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

    # Ângulo inicial de rotação e fator de escala
    theta = 0
    s = 1  # Fator de escala inicial
    delta_s = 0.01  # Taxa de variação da escala
    rot_speed = 0.03

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

        # Atualiza o ângulo de rotação e o fator de escala a cada frame
        tecla = cv.waitKey(1)

        if tecla == ord('r'):
            rot_speed += 0.01  
        elif tecla == ord('d'):
            rot_speed = max(0.01, rot_speed - 0.01) 
        elif tecla == ord('n'):
            rot_speed = 0.03

        theta += rot_speed  # Ajuste o valor para controlar a velocidade da rotação
        s += delta_s  # Atualiza o fator de escala

        # Inverte a direção da escala quando atinge limites
        if s >= 1.5 or s <= 0.5:
            delta_s *= -1  # Inverte a direção da escala

        # Calcula a matriz de rotação e escala para o ângulo e fator de escala atuais
        R = rotacao(theta)
        S = escala(s)

        # Matriz de transformação composta (translada para o centro, aplica escala e rotação, e volta para a posição original)
        Y = T_origem_para_centro @ S @ R @ T_centro_para_origem

        # Gera os índices dos pixels da imagem
        Xd = criar_indices(0, image.shape[0], 0, image.shape[1])
        Xd = np.vstack((Xd, np.ones(Xd.shape[1])))

        # Aplica a transformação
        X = np.linalg.inv(Y) @ Xd
        X = X.astype(int)
        Xd = Xd.astype(int)
        
        # Aplica o clipping nos índices ANTES de acessar a imagem
        X[0, :] = np.clip(X[0, :], 0, image.shape[0] - 1)  # Limita no eixo vertical
        X[1, :] = np.clip(X[1, :], 0, image.shape[1] - 1)  # Limita no eixo horizontal

        # Faz a atribuição dos pixels transformados
        image_[Xd[0, :], Xd[1, :], :] = image[X[0, :], X[1, :], :]

        # Mostra a imagem na tela
        cv.imshow('Minha Imagem!', image_)

        # Sai do loop se 'q' for pressionado
        if tecla == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
