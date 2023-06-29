package org.apache.spark.ml.regression

import org.apache.spark.ml.linalg.{Matrices, Vector, Vectors}
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.{col, udf}
import org.scalatest.flatspec._
import org.scalatest.matchers._


import scala.util.Random

class LinearRegressionTest extends AnyFlatSpec with should.Matchers with WithSpark {

    val epsilon = 0.0001
    lazy val data: DataFrame = LinearRegrTest._data
    lazy val vectors: Seq[Vector] = LinearRegrTest._vectors
    val UserPredict: UserDefinedFunction = LinearRegrTest._Userpredict


    "Model" should "predict input data" in {
        val model: LRModel = new LRModel(
            LRweights = Vectors.dense(1.5, 0.3, -0.7),
        )

        validateModel(model.transform(data))
    }

    "Estimator" should "produce a LinearRegressionModel with non-NaN coefficients" in {
        val estimator = new LinearRegression_().numIter(10000)

        import sqlContext.implicits._

        val randomData = Matrices.rand(100000, 3, Random.self).rowIter.toSeq.map(x => Tuple1(x)).toDF("features")

        val data = randomData.withColumn("label", UserPredict(col("features")))
        val LinReg = estimator.fit(data)

        val weights = LinReg.LRweights.toArray

        weights.foreach(coef => coef.isNaN should be(false))
    }

    "Estimator" should "make right predictions" in {
        val estimator = new LinearRegression_().numIter(10000)

        import sqlContext.implicits._

        val randomData = Matrices.rand(100000, 3, Random.self).rowIter.toSeq.map(x => Tuple1(x)).toDF("features")

        val data = randomData.withColumn("label", UserPredict(col("features")))
        val LinReg = estimator.fit(data)

        println("Obtained coefficients:")
        println(s" - Coefficient 1: ${LinReg.LRweights(0)}")
        println(s" - Coefficient 2: ${LinReg.LRweights(1)}")
        println(s" - Coefficient 3: ${LinReg.LRweights(2)}")

        LinReg.LRweights(0) should be(1.5 +- epsilon)
        LinReg.LRweights(1) should be(0.3 +- epsilon)
        LinReg.LRweights(2) should be(-0.7 +- epsilon)
    }

    private def validateModel(data: DataFrame): Unit = {
        val y_pred = data.collect().map(_.getAs[Double](1))

        y_pred.length should be(2)

        y_pred(0).isNaN should be(false)
        y_pred(1).isNaN should be(false)

        y_pred(0) should be(21.05 +- epsilon)
        y_pred(1) should be(-1.6 +- epsilon)
    }

}



object LinearRegrTest extends WithSpark {
    lazy val _vectors: Seq[Vector] = Seq(
        Vectors.dense(13.5, 12, 4),
        Vectors.dense(-1, 2, 1)
    )

    lazy val _data: DataFrame = {
        import sqlContext.implicits._
        _vectors.map(x => Tuple1(x)).toDF("features")
    }

    val _Userpredict: UserDefinedFunction = udf { features: Any =>
        val arr = features.asInstanceOf[Vector].toArray
        val coefficients = Array(1.5, 0.3, -0.7)

        require(arr.length == coefficients.length, "Number of features and coefficients should match")

        val prediction = (arr, coefficients).zipped.map(_ * _).sum
        prediction
    }
}
