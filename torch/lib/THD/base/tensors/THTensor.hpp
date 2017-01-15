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

  virtual thd::Type type() const override;
  virtual std::unique_ptr<Tensor> newTensor() const override;

  virtual Tensor *newWithSize1d(long size0_) override;
  virtual Tensor *newWithSize2d(long size0_, long size1_) override;
  virtual Tensor *newWithSize3d(long size0_, long size1_, long size2_) override;
  virtual Tensor *newWithSize4d(long size0_, long size1_, long size2_, long size3_) override;
  virtual Tensor *newWithStorage(Storage *storage_, ptrdiff_t storageOffset_, StorageScalarInterface<long> *size_, StorageScalarInterface<long> *stride_) override;
  virtual Tensor *newWithStorage1d(Storage *storage_, ptrdiff_t storageOffset_, long size0_, long stride0_) override;
  virtual Tensor *newWithStorage2d(Storage *storage_, ptrdiff_t storageOffset_, long size0_, long stride0_, long size1_, long stride1_) override;
  virtual Tensor *newWithStorage3d(Storage *storage_, ptrdiff_t storageOffset_, long size0_, long stride0_, long size1_, long stride1_, long size2_, long stride2_) override;
  virtual Tensor *newWithStorage4d(Storage *storage_, ptrdiff_t storageOffset_, long size0_, long stride0_, long size1_, long stride1_, long size2_, long stride2_, long size3_, long stride3_) override;
  virtual Tensor *newClone(Tensor *self) override;
  virtual Tensor *newContiguous(Tensor *tensor) override;
  virtual Tensor *newSelect(Tensor *tensor, int dimension_, long sliceIndex_) override;
  virtual Tensor *newNarrow(Tensor *tensor, int dimension_, long firstIndex_, long size_) override;
  virtual Tensor *newTranspose(Tensor *tensor, int dimension1_, int dimension2_) override;
  virtual Tensor *newUnfold(Tensor *tensor, int dimension_, long size_, long step_) override;
  virtual void narrow(Tensor *self, Tensor *src, int dimension_, long firstIndex_, long size_) override;
  virtual void select(Tensor *self, Tensor *src, int dimension_, long sliceIndex_) override;
  virtual void transpose(Tensor *self, Tensor *src, int dimension1_, int dimension2_) override;
  virtual void unfold(Tensor *self, Tensor *src, int dimension_, long size_, long step_) override;
  virtual void squeeze(Tensor *self, Tensor *src) override;
  virtual void squeeze1d(Tensor *self, Tensor *src, int dimension_) override;
  virtual int isContiguous(const Tensor *self) override;
  virtual int isSameSizeAs(const Tensor *self, const Tensor *src) override;
  virtual int isSetTo(const Tensor *self, const Tensor *src) override;
  virtual int isSize(const Tensor *self, const StorageScalarInterface<long> *dims) override;
  virtual ptrdiff_t nElement(const Tensor *self) override;
  virtual void gesv(Tensor *rb_, Tensor *ra_, Tensor *b_, Tensor *a_) override;
  virtual void trtrs(Tensor *rb_, Tensor *ra_, Tensor *b_, Tensor *a_, const char *uplo, const char *trans, const char *diag) override;
  virtual void gels(Tensor *rb_, Tensor *ra_, Tensor *b_, Tensor *a_) override;
  virtual void syev(Tensor *re_, Tensor *rv_, Tensor *a_, const char *jobz, const char *uplo) override;
  virtual void geev(Tensor *re_, Tensor *rv_, Tensor *a_, const char *jobvr) override;
  virtual void gesvd(Tensor *ru_, Tensor *rs_, Tensor *rv_, Tensor *a, const char *jobu) override;
  virtual void gesvd2(Tensor *ru_, Tensor *rs_, Tensor *rv_, Tensor *ra_, Tensor *a, const char *jobu) override;
  virtual void getri(Tensor *ra_, Tensor *a) override;
  virtual void potrf(Tensor *ra_, Tensor *a, const char *uplo) override;
  virtual void potrs(Tensor *rb_, Tensor *b_, Tensor *a_,  const char *uplo) override;
  virtual void potri(Tensor *ra_, Tensor *a, const char *uplo) override;
  virtual void qr(Tensor *rq_, Tensor *rr_, Tensor *a) override;
  virtual void geqrf(Tensor *ra_, Tensor *rtau_, Tensor *a) override;
  virtual void orgqr(Tensor *ra_, Tensor *a, Tensor *tau) override;
  virtual void ormqr(Tensor *ra_, Tensor *a, Tensor *tau, Tensor *c, const char *side, const char *trans) override;
  virtual void pstrf(Tensor *ra_, TensorScalarInterface<int> *rpiv_, Tensor* a, const char* uplo, scalar_type tol) override;
  virtual void fill(Tensor *r_, scalar_type value) override;
  virtual void zero(Tensor *r_) override;
  virtual void maskedFill(Tensor *tensor, TensorScalarInterface<unsigned char> *mask, scalar_type value) override;
  virtual void maskedCopy(Tensor *tensor, TensorScalarInterface<unsigned char> *mask, Tensor* src) override;
  virtual void maskedSelect(Tensor *tensor, Tensor* src, TensorScalarInterface<unsigned char> *mask) override;
  virtual void nonzero(TensorScalarInterface<long> *subscript, Tensor *tensor) override;
  virtual void indexSelect(Tensor *tensor, Tensor *src, int dim, TensorScalarInterface<long> *index) override;
  virtual void indexCopy(Tensor *tensor, int dim, TensorScalarInterface<long> *index, Tensor *src) override;
  virtual void indexAdd(Tensor *tensor, int dim, TensorScalarInterface<long> *index, Tensor *src) override;
  virtual void indexFill(Tensor *tensor, int dim, TensorScalarInterface<long> *index, scalar_type val) override;

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
