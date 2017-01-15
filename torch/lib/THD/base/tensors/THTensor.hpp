#pragma once

#include <TH/TH.h>

// We're defining THTensor as a custom class
#undef THTensor
#define THRealTensor TH_CONCAT_3(TH,Real,Tensor)

#include "../Tensor.hpp"
#include "../Traits.hpp"

namespace thd {

template<typename real>
struct th_tensor_traits {};

#include "base/tensors/generic/THTensor.hpp"
#include <TH/THGenerateAllTypes.h>

} // namespace thd

#include "../storages/THStorage.hpp"

namespace thd {

template<typename real>
struct THTensor : public interface_traits<real>::tensor_interface_type {
  friend class THTensor<unsigned char>;
  friend class THTensor<char>;
  friend class THTensor<short>;
  friend class THTensor<int>;
  friend class THTensor<long>;
  friend class THTensor<float>;
  friend class THTensor<double>;

private:
  using interface_type = typename interface_traits<real>::tensor_interface_type;
public:
  using tensor_type = typename th_tensor_traits<real>::tensor_type;
  using scalar_type = typename interface_type::scalar_type;
  using long_range = Tensor::long_range;

  THTensor();
  THTensor(tensor_type *wrapped);
  virtual ~THTensor();

  THTensor(long size0_);
  THTensor(long size0_, long size1_);
  THTensor(long size0_, long size1_, long size2_);
  THTensor(long size0_, long size1_, long size2_, long size3_);
  THTensor(const Storage& storage_, ptrdiff_t storageOffset_, THLongStorage *size_, THLongStorage *stride_);
  THTensor(const Storage& storage_, ptrdiff_t storageOffset_, long size0_, long stride0_);
  THTensor(const Storage& storage_, ptrdiff_t storageOffset_, long size0_, long stride0_, long size1_, long stride1_);
  THTensor(const Storage& storage_, ptrdiff_t storageOffset_, long size0_, long stride0_, long size1_, long stride1_, long size2_, long stride2_);
  THTensor(const Storage& storage_, ptrdiff_t storageOffset_, long size0_, long stride0_, long size1_, long stride1_, long size2_, long stride2_, long size3_, long stride3_);

  virtual THTensor* clone() const override;
  virtual THTensor* clone_shallow() override;

  virtual int nDim() const override;
  virtual long_range sizes() const override;
  virtual long_range strides() const override;
  virtual const long* rawSizes() const override;
  virtual const long* rawStrides() const override;
  virtual std::size_t storageOffset() const override;
  virtual std::size_t elementSize() const override;
  virtual long long numel() const override;
  virtual bool isContiguous() const override;
  virtual void* data() override;
  virtual const void* data() const override;
  virtual THTensor& retain() override;
  virtual THTensor& free() override;

  virtual THTensor& resize(const std::initializer_list<long>& new_size) override;
  virtual THTensor& resize(const std::vector<long>& new_size) override;
  virtual THTensor& resize(THLongStorage *size,
                           THLongStorage *stride) override;
  virtual THTensor& resizeAs(const Tensor& src) override;
  virtual THTensor& set(const Tensor& src) override;
  virtual THTensor& setStorage(const Storage& storage,
                             ptrdiff_t storageOffset,
                             THLongStorage *size,
                             THLongStorage *stride) override;

  virtual THTensor& narrow(const Tensor& src, int dimension,
                           long firstIndex, long size) override;
  virtual THTensor& select(const Tensor& src, int dimension,
                           long sliceIndex) override;
  virtual THTensor& transpose(const Tensor& src, int dimension1,
                              int dimension2) override;
  virtual THTensor& unfold(const Tensor& src, int dimension,
                           long size, long step) override;

  virtual THTensor& fill(scalar_type value) override;

  virtual THTensor& gather(const Tensor& src, int dimension, const Tensor& index) override;
  virtual THTensor& scatter(int dimension, const Tensor& index, const Tensor& src) override;
  virtual THTensor& scatterFill(int dimension, const Tensor& index, scalar_type value) override;
  virtual scalar_type dot(const Tensor& source) override;
  virtual scalar_type minall() override;
  virtual scalar_type maxall() override;
  virtual scalar_type sumall() override;
  virtual scalar_type prodall() override;
  virtual THTensor& neg(const Tensor& src) override;
  virtual THTensor& cinv(const Tensor& src) override;
  virtual THTensor& add(const Tensor& src, scalar_type value) override;
  virtual THTensor& sub(const Tensor& src, scalar_type value) override;
  virtual THTensor& mul(const Tensor& src, scalar_type value) override;
  virtual THTensor& div(const Tensor& src, scalar_type value) override;
  virtual THTensor& fmod(const Tensor& src, scalar_type value) override;
  virtual THTensor& remainder(const Tensor& src, scalar_type value) override;
  virtual THTensor& clamp(const Tensor& src, scalar_type min_value, scalar_type max_value) override;
  virtual THTensor& cadd(const Tensor& src1, scalar_type value, const Tensor& src2) override;
  virtual THTensor& csub(const Tensor& src1, scalar_type value, const Tensor& src2) override;
  virtual THTensor& cmul(const Tensor& src1, const Tensor& src2) override;
  virtual THTensor& cpow(const Tensor& src1, const Tensor& src2) override;
  virtual THTensor& cdiv(const Tensor& src1, const Tensor& src2) override;
  virtual THTensor& cfmod(const Tensor& src1, const Tensor& src2) override;
  virtual THTensor& cremainder(const Tensor& src1, const Tensor& src2) override;
  virtual THTensor& addcmul(const Tensor& src1, scalar_type value, const Tensor& src2, const Tensor& src3) override;
  virtual THTensor& addcdiv(const Tensor& src1, scalar_type value, const Tensor& src2, const Tensor& src3) override;
  virtual THTensor& addmv(scalar_type beta, const Tensor& src, scalar_type alpha, const Tensor& mat, const Tensor& vec) override;
  virtual THTensor& addmm(scalar_type beta, const Tensor& src, scalar_type alpha, const Tensor& mat1, const Tensor& mat2) override;
  virtual THTensor& addr(scalar_type beta, const Tensor& src, scalar_type alpha, const Tensor& vec1, const Tensor& vec2) override;
  virtual THTensor& addbmm(scalar_type beta, const Tensor& src, scalar_type alpha, const Tensor& batch1, const Tensor& batch2) override;
  virtual THTensor& baddbmm(scalar_type beta, const Tensor& src, scalar_type alpha, const Tensor& batch1, const Tensor& batch2) override;
  virtual THTensor& match(const Tensor& m1, const Tensor& m2, scalar_type gain) override;
  virtual THTensor& max(const Tensor& indices_, const Tensor& src, int dimension) override;
  virtual THTensor& min(const Tensor& indices_, const Tensor& src, int dimension) override;
  virtual THTensor& kthvalue(const Tensor& indices_, const Tensor& src, long k, int dimension) override;
  virtual THTensor& mode(const Tensor& indices_, const Tensor& src, int dimension) override;
  virtual THTensor& median(const Tensor& indices_, const Tensor& src, int dimension) override;
  virtual THTensor& sum(const Tensor& src, int dimension) override;
  virtual THTensor& prod(const Tensor& src, int dimension) override;
  virtual THTensor& cumsum(const Tensor& src, int dimension) override;
  virtual THTensor& cumprod(const Tensor& src, int dimension) override;
  virtual THTensor& sign(const Tensor& source) override;
  virtual scalar_type trace() override;
  virtual THTensor& cross(const Tensor& src1, const Tensor& src2, int dimension) override;
  virtual THTensor& cmax(const Tensor& src1, const Tensor& src2) override;
  virtual THTensor& cmin(const Tensor& src1, const Tensor& src2) override;
  virtual THTensor& cmaxValue(const Tensor& src, scalar_type value) override;
  virtual THTensor& cminValue(const Tensor& src, scalar_type value) override;

  // XXX: Mateusz

  virtual THTensor* newClone() const override;

  // virtual THTensor& squeeze(const Tensor& src) override;
  // virtual THTensor& squeeze1d(const Tensor& src, int dimension_) override;
  // virtual int isContiguous() override;
  // virtual int isSameSizeAs(const Tensor& src) override;
  // virtual int isSetTo(const Tensor& src) override;
  // virtual int isSize(const StorageScalarInterface<long>& dims) override;
  // virtual ptrdiff_t nElement() override;
  // virtual void gesv(const Tensor& rb_, const Tensor& ra_, const Tensor& b_, const Tensor& a_) override;
  // virtual void trtrs(const Tensor& rb_, const Tensor& ra_, const Tensor& b_, const Tensor& a_, const char *uplo, const char *trans, const char *diag) override;
  // virtual void gels(const Tensor& rb_, const Tensor& ra_, const Tensor& b_, const Tensor& a_) override;
  // virtual void syev(const Tensor& re_, const Tensor& rv_, const Tensor& a_, const char *jobz, const char *uplo) override;
  // virtual void geev(const Tensor& re_, const Tensor& rv_, const Tensor& a_, const char *jobvr) override;
  // virtual void gesvd(const Tensor& ru_, const Tensor& rs_, const Tensor& rv_, const Tensor& a, const char *jobu) override;
  // virtual void gesvd2(const Tensor& ru_, const Tensor& rs_, const Tensor& rv_, const Tensor& ra_, const Tensor& a, const char *jobu) override;
  // virtual void getri(const Tensor& ra_, const Tensor& a) override;
  // virtual void potrf(const Tensor& ra_, const Tensor& a, const char *uplo) override;
  // virtual void potrs(const Tensor& rb_, const Tensor& b_, const Tensor& a_,  const char *uplo) override;
  // virtual void potri(const Tensor& ra_, const Tensor& a, const char *uplo) override;
  // virtual void qr(const Tensor& rq_, const Tensor& rr_, Tensor *a) override;
  // virtual void geqrf(const Tensor& ra_, const Tensor& rtau_, const Tensor& a) override;
  // virtual void orgqr(const Tensor& ra_, const Tensor& a, const Tensor& tau) override;
  // virtual void ormqr(const Tensor& ra_, const Tensor& a, const Tensor& tau, const Tensor& c, const char *side, const char *trans) override;
  // virtual THTensor& zero() override;
  // virtual THTensor* newWithSize2d(long size0_, long size1_) override;
  // virtual THTensor* newWithSize3d(long size0_, long size1_, long size2_) override;
  // virtual THTensor* newWithSize4d(long size0_, long size1_, long size2_, long size3_) override;
  // virtual THTensor* newWithStorage(const Storage& storage_, ptrdiff_t storageOffset_, const StorageScalarInterface<long>& size_, const StorageScalarInterface<long>& stride_) override;
  // virtual THTensor* newWithStorage1d(const Storage& storage_, ptrdiff_t storageOffset_, long size0_, long stride0_) override;
  // virtual THTensor* newWithStorage2d(const Storage& storage_, ptrdiff_t storageOffset_, long size0_, long stride0_, long size1_, long stride1_) override;
  // virtual THTensor* newWithStorage3d(const Storage& storage_, ptrdiff_t storageOffset_, long size0_, long stride0_, long size1_, long stride1_, long size2_, long stride2_) override;
  // virtual THTensor* newWithStorage4d(const Storage& storage_, ptrdiff_t storageOffset_, long size0_, long stride0_, long size1_, long stride1_, long size2_, long stride2_, long size3_, long stride3_) override;
  // virtual THTensor* newClone() override;
  // virtual THTensor* newContiguous() override;
  // virtual THTensor* newSelect(int dimension_, long sliceIndex_) override;
  // virtual THTensor* newNarrow(int dimension_, long firstIndex_, long size_) override;
  // virtual THTensor* newTranspose(int dimension1_, int dimension2_) override;
  // virtual THTensor* newUnfold(int dimension_, long size_, long step_) override;
  // virtual THTensor& pstrf(const Tensor& ra_, const TensorScalarInterface<int>& rpiv_, const Tensor*& a, const char uplo, scalar_type tol) override;
  // virtual THTensor& fill(const Tensor& r_, scalar_type value) override;
  // virtual THTensor& maskedFill(const Tensor& tensor, const TensorScalarInterface<unsigned char>& mask, scalar_type value) override;
  // virtual THTensor<scalar_type> maskedCopy(const TensorScalarInterface<unsigned char>& mask, const Tensor& src) override;
  // virtual THTensor<scalar_type> maskedSelect(const Tensor& src, const TensorScalarInterface<unsigned char>& mask) override;
  // virtual void nonzero(const TensorScalarInterface<long>& subscript, const Tensor& tensor) override; // TODO: Is this signature fine?
  // virtual THTensor<scalar_type> indexSelect(const Tensor& src, int dim, const TensorScalarInterface<long>& index) override;
  // virtual THTensor<scalar_type> indexCopy(int dim, const TensorScalarInterface<long>& index, const Tensor& src) override;
  // virtual THTensor<scalar_type> indexAdd(int dim, const TensorScalarInterface<long>& index, const Tensor& src) override;
  // virtual THTensor<scalar_type> indexFill(int dim, const TensorScalarInterface<long>& index, scalar_type val) override;

  virtual thd::Type type() const override;
  virtual std::unique_ptr<Tensor> newTensor() const override;

private:
  template<typename iterator>
  THTensor& resize(const iterator& begin, const iterator& end);
  template<typename iterator>
  THTensor& resize(const iterator& size_begin, const iterator& size_end,
                   const iterator& stride_begin, const iterator& stride_end);

protected:
  tensor_type *tensor;
};

}
