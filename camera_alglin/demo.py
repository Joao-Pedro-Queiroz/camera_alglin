import numpy as np
import cv2 as cv


def run():
    # Essa função abre a câmera. Depois desta linha, a luz de câmera (se seu computador tiver) deve ligar.
    cap = cv.VideoCapture(0)

    # Aqui, defino a largura e a altura da imagem com a qual quero trabalhar.
    # Dica: imagens menores precisam de menos processamento!!!
    width = 320
    height = 240

    #Matrizes de transformação
    T = np.array([[1, 0, height/2], [0, 1, width/2], [0, 0, 1]])
    R = np.array([[0.7, -0.7, 0], [0.7, 0.7, 0], [0, 0, 1]])
    X = np.linalg.inv(T) @ R @ T

    # Talvez o programa não consiga abrir a câmera. Verifique se há outros dispositivos acessando sua câmera!
    if not cap.isOpened():
        print("Não consegui abrir a câmera!")
        exit()

    # Esse loop é igual a um loop de jogo: ele encerra quando apertamos 'q' no teclado.
    while True:
        # Captura um frame da câmera
        ret, frame = cap.read()

        # A variável `ret` indica se conseguimos capturar um frame
        if not ret:
            print("Não consegui capturar frame!")
            break

        # Mudo o tamanho do meu frame para reduzir o processamento necessário
        # nas próximas etapas
        frame = cv.resize(frame, (width,height), interpolation =cv.INTER_AREA)
        frame_trnsformado = X @ frame
        frame_trnsformado = frame_trnsformado.astype(int)

        # Troque este código pelo seu código de filtragem de pixels
        frame_trnsformado[0,:] = np.clip(frame_trnsformado[0,:], 0, frame.shape[0])
        frame_trnsformado[1,:] = np.clip(frame_trnsformado[1,:], 0, frame.shape[1])

        # A variável image é um np.array com shape=(width, height, colors)
        image = np.array(frame_trnsformado).astype(float)/255

        # Agora, mostrar a imagem na tela!
        cv.imshow('Minha Imagem!', image)
        
        # Se aperto 'q', encerro o loop
        if cv.waitKey(1) == ord('q'):
            break

    # Ao sair do loop, vamos devolver cuidadosamente os recursos ao sistema!
    cap.release()
    cv.destroyAllWindows()