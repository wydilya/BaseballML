//
//  Predictor.swift
//  BaseballML
//
//  Created by Ilya Zelkin on 02.07.2022.
//

import Foundation
import Vision

typealias ThrowingClassifier = BaseballML_1

protocol PredictedDelegate: AnyObject {
    func predictor(_ predictor: Predictor, didFindNewRecognizedPoints points: [CGPoint])
    func predictor(_ predictor: Predictor, didLabelAction action: String, with confidence: Double)
}

class Predictor {
    
    weak var delegate: PredictedDelegate?
    
    var predictionWindowSize = 30
    var posesWindow: [VNHumanBodyPoseObservation] = []
    
    init() {
        posesWindow.reserveCapacity(predictionWindowSize)
    }
    
    func estimation(sampleBuffer: CMSampleBuffer) {
        let requestHandler = VNImageRequestHandler(cmSampleBuffer: sampleBuffer, orientation: .up)
        
        let request = VNDetectHumanBodyPoseRequest(completionHandler: bodyPoseHandler)
        
        do {
            try requestHandler.perform([request])
        } catch {
            print("Unable to perform the request, with error: \(error)")
        }
    }
    
    func bodyPoseHandler(request: VNRequest, error: Error?) {
        guard let observation = request.results as? [VNHumanBodyPoseObservation] else { return }
        
        observation.forEach {
            processObservation($0)
        }
        
        if let result = observation.first {
            storeObservation(result)
            
            labelActionType()
        }
    }
    
    func labelActionType() {
        guard let throwingClassifier = try? ThrowingClassifier(configuration: MLModelConfiguration()),
              let poseMultiArray = prepareInputWithObservations(posesWindow),
              let predictions = try? throwingClassifier.prediction(poses: poseMultiArray) else { return }
        
        let label = predictions.label
        let confidence = predictions.labelProbabilities[label] ?? 0
        
        delegate?.predictor(self, didLabelAction: label, with: confidence)
    }
    
    func prepareInputWithObservations(_ observations: [VNHumanBodyPoseObservation]) -> MLMultiArray? {
        let numAvailableFrames = observations.count
        let observationsNeeded = 30
        var multiArrayBuffer = [MLMultiArray]()
        
        for frameIndex in 0..<min(numAvailableFrames, observationsNeeded) {
            let pose = observations[frameIndex]
            do {
                let oneFrameMultiArrayBuffer = try pose.keypointsMultiArray()
                multiArrayBuffer.append(oneFrameMultiArrayBuffer)
            } catch {
                continue
            }
        }
        
        if numAvailableFrames < observationsNeeded {
            for _ in 0..<(observationsNeeded - numAvailableFrames) {
                do {
                    let oneFrameMultiArray = try MLMultiArray(shape: [1, 3, 18], dataType: .double)
                    try resetMultiArray(oneFrameMultiArray)
                    multiArrayBuffer.append(oneFrameMultiArray)
                } catch {
                    continue
                }
            }
        }
        
        return MLMultiArray(concatenating: [MLMultiArray](multiArrayBuffer), axis: 0, dataType: .float)
    }
    
    func resetMultiArray(_ predictionWindow: MLMultiArray, with value: Double = 0.0) throws {
        let pointer = try UnsafeMutableBufferPointer<Double>(predictionWindow)
        pointer.initialize(repeating: value)
    }
    
    func storeObservation(_ observation: VNHumanBodyPoseObservation) {
        if posesWindow.count >= predictionWindowSize {
            posesWindow.removeFirst()
        }
        
        posesWindow.append(observation)
    }
    
    func processObservation(_ observation: VNHumanBodyPoseObservation) {
        do {
            let recognizedPoints = try observation.recognizedPoints(forGroupKey: .all)
            
            let displayedPoints = recognizedPoints.map {
                CGPoint(x: $0.value.x, y: 1 - $0.value.y)
            }
            
            delegate?.predictor(self, didFindNewRecognizedPoints: displayedPoints)
        } catch {
            print("error finding recognizedPoints")
        }
    }
}
