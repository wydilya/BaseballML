//
//  ViewController.swift
//  BaseballML
//
//  Created by Ilya Zelkin on 02.07.2022.
//

import UIKit
import AVFoundation
//import AudioToolbox

class ViewController: UIViewController {
    
    let videoCapture = VideoCapture()
    var previewLayer: AVCaptureVideoPreviewLayer?
    
    var pointsLayer = CAShapeLayer()
    
    var isThrowDetected = false

    override func viewDidLoad() {
        super.viewDidLoad()
        
        setupVideoPreview()
        
        videoCapture.predictor.delegate = self
    }

    func setupVideoPreview() {
        videoCapture.startCaptureSession()
        previewLayer = AVCaptureVideoPreviewLayer(session: videoCapture.captureSession)
        
        guard let previewLayer = previewLayer else { return }
        
        view.layer.addSublayer(previewLayer)
        previewLayer.frame = view.frame
        
        view.layer.addSublayer(pointsLayer)
        pointsLayer.frame = view.frame
        pointsLayer.strokeColor = UIColor.green.cgColor
    }

}

extension ViewController: PredictedDelegate {
    func predictor(_ predictor: Predictor, didLabelAction action: String, with confidence: Double) {
        if action == "Throw" && confidence > 0.95 && isThrowDetected == false  {
            print("Throw Detected")
            isThrowDetected = true
            
            DispatchQueue.main.asyncAfter(deadline: .now() + 3) {
                self.isThrowDetected = true
            }
            
            DispatchQueue.main.async {
                AudioServicesPlayAlertSound(SystemSoundID(1322))
            }
        }
    }
    
    func predictor(_ predictor: Predictor, didFindNewRecognizedPoints points: [CGPoint]) {
        guard let previewLayer = previewLayer else { return }
        
        let convertedPoints = points.map {
            previewLayer.layerPointConverted(fromCaptureDevicePoint: $0)
        }
        
        let combinedPath = CGMutablePath()
        
        for point in convertedPoints {
            let dotPath = UIBezierPath(ovalIn: CGRect(x: point.x, y: point.y, width: 10, height: 10))
            combinedPath.addPath(dotPath.cgPath)
        }
        pointsLayer.path = combinedPath
        
        DispatchQueue.main.async {
            self.pointsLayer.didChangeValue(for: \.path)
        }
    }
}

